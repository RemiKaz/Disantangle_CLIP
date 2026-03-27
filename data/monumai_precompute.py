import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import kornia
import clip
import numpy as np
from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from clip_text_span.prs_hook import hook_prs_logger
import os
import random
from PIL import Image
import glob

class MonumaiPrecomputeDataset(Dataset):
    def __init__(
        self,
        root,
        split='train',
        model_name='ViT-B-16',
        pretrained='laion2b_s34b_b88k',
        device='cuda',
        list_concepts=None,
        restrict_samples=None,
        disantangled_version=False,
    ):
        """
        Args:
            root (str): Path to the Monumai dataset.
            split (str): 'train', 'val', or 'test'.
            model_name (str): CLIP model name.
            pretrained (str): Pretrained weights for CLIP.
            device (str): Device to load tensors.
            pth_metadata (str): Path to metadata file for Monumai classes.
            pth_metadata_parts (str): Path to metadata file for parts mapping.
            list_concepts (list): List of concepts for disentangled version.
            disantangled_version (bool/str): Whether to use disentangled version.
        """
        self.device = device
        self.split = split
        self.batch_size = 2  # Batch size for precomputing embeddings
        self.disantangled_version = disantangled_version
        self.root = root
        
        # Load CLIP model and preprocessor
        if 'laion' in pretrained:
            self.model_clip, _, self.preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = get_tokenizer(model_name)
            self.model_clip.to(self.device)
            self.model_clip.eval()
        else:
            raise NotImplementedError("Only LAION CLIP is supported for now.")

        # Load dataset
        self.list_classes = ['Baroque', 'Gothic', 'Hispanic-Muslim', 'Renaissance']

        self.dict_concepts = {
            'Hispanic-Muslim': ['Horseshoe arch', 'Lobed arch', 'Flat arch'],
            'Gothic': ['Pointed arch', 'Ogee arch', 'Trefoil arch', 'Gothic pinnacle'],
            'Renaissance': ['Triangular pediment', 'Segmental pediment', 'Serliana', 'Porthole', 'Lintelled doorway', 'Rounded arch'],
            'Baroque': ['Porthole', 'Lintelled doorway', 'Rounded arch', 'Broken pediment', 'Solomonic column']
        }
            
        # Store concepts and classes
        self.disantangled_version = disantangled_version
        
        self.list_concepts = []
        
        for class_name in self.list_classes:
            self.list_concepts += self.dict_concepts[class_name]

        # Initialize disentangled version parameters
        if self.disantangled_version:
            self.method_iou, self.strategy, self.type_concept = disantangled_version.split('-')
            self.prs = hook_prs_logger(self.model_clip, self.device)
            self.image_size = 224 if model_name.startswith('ViT-B') else 256
            self.patch_size = 16 if model_name.startswith('ViT-B') else 14
            self.weights = torch.tensor(np.mean(
                np.load(f'weights/weights_monumai_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'),
                axis=0
            )).to(self.device)

        # Precompute text embeddings
        if self.list_concepts is not None:
            tokens = clip.tokenize(self.list_concepts).to(self.device)
            with torch.no_grad():
                self.text_embeddings = self.model_clip.encode_text(tokens).float()

        """Load and process images and annotations from the Monumai dataset."""
        img_path_pattern = os.path.join(self.root, '*', '*.jpg')
        xml_path_pattern = os.path.join(self.root, '*', 'xml', '*.xml')
        self.paths_images = sorted(glob.glob(img_path_pattern))
        self.paths_annotations = sorted(glob.glob(xml_path_pattern))

        random.seed(667)
        fused_list = list(zip(self.paths_images, self.paths_annotations))
        random.shuffle(fused_list)
        self.paths_images, self.paths_annotations = zip(*fused_list)
        self.List_data_image = []

        img_paths_train, img_paths_val, img_paths_test, ann_paths_train, ann_paths_val, ann_paths_test = self.split_data(
            self.paths_images, self.paths_annotations)

        if self.split == 'train':
            self.paths_images, _ = img_paths_train, ann_paths_train
        elif self.split == 'val':
            self.paths_images, _ = img_paths_val, ann_paths_val
        elif self.split == 'test':
            self.paths_images, _ = img_paths_test, ann_paths_test
        else:
            raise ValueError("Invalid phase specified. Choose from 'train', 'val', or 'test'.")

        self.list_labels = []
        
        for img_path in self.paths_images:
            self.list_labels.append(self.list_classes.index(self.label_from_path(img_path)))

        # Precompute image embeddings
        self.image_embeddings = self._precompute_image_embeddings()

    def split_data(self, img_paths, annotation_paths, split_seed=667, train_ratio=0.6, val_ratio=0.2):
        """Split the dataset into training, validation, and test sets."""
        combined = list(zip(img_paths, annotation_paths))
        random.seed(split_seed)
        random.shuffle(combined)
        img_paths_shuffled, annotation_paths_shuffled = zip(*combined)
        total_size = len(img_paths_shuffled)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        img_paths_train = img_paths_shuffled[:train_size]
        annotation_paths_train = annotation_paths_shuffled[:train_size]
        img_paths_val = img_paths_shuffled[train_size:train_size + val_size]
        annotation_paths_val = annotation_paths_shuffled[train_size:train_size + val_size]
        img_paths_test = img_paths_shuffled[train_size + val_size:]
        annotation_paths_test = annotation_paths_shuffled[train_size + val_size:]
        return img_paths_train, img_paths_val, img_paths_test, annotation_paths_train, annotation_paths_val, annotation_paths_test

    def label_from_path(self, path):
        """Extract label from image path."""
        for label in self.list_classes:
            if label in path:
                return label
        raise ValueError(f"No valid label found in path: {path}")

    def _precompute_image_embeddings(self, batch_size=32, num_workers=8):
        """Precompute image embeddings for the entire dataset with batching & GPU."""
        
        all_embeddings = []

        # load the embeddings if they exist
        if self.disantangled_version is not False:
            if os.path.exists(f'embeds/embeds_monumai_{self.disantangled_version}_{self.split}.npy'):
                all_embeddings = np.load(f'embeds/embeds_monumai_{self.disantangled_version}_{self.split}.npy')
                return torch.tensor(all_embeddings)

        elif self.disantangled_version is False:
            text_inputs = torch.cat(
                [clip.tokenize(c) for c in self.list_concepts]
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model_clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            if os.path.exists(f'embeds/embeds_monumai_{self.split}.npy'):
                all_embeddings = np.load(f'embeds/embeds_monumai_{self.split}.npy')
                return torch.tensor(all_embeddings)

        with torch.no_grad():
            for path_image in tqdm(self.paths_images, desc="Precomputing image embeddings"):
                
                # Warning and pass is path do not exist
                if os.path.exists(os.path.join(self.root, path_image)):
                    # Batch image preprocessing
                    items = [Image.open(path_image).convert("RGB")]
                else:
                    print(f"Warning: Image {path_image} does not exist")
                    continue
            
                if isinstance(items, torch.Tensor):
                    images = items.to(self.device, non_blocking=True)
                else:
                    # Fallback (not recommended if dataset returns PILs)
                    images = torch.stack([self.preprocess(img) for img in items]).to(self.device)

                if self.disantangled_version and self.weights is not None:
                    self.prs.reinit()

                    # Encode images in batch
                    image_features = self.model_clip.encode_image(
                        images,
                        attn_method="head",
                        normalize=False
                    )   # shape: [B, ...]

                    # Encode all concepts ONCE outside loop
                    if not hasattr(self, "_cached_text_features"):
                        text_inputs = torch.cat(
                            [clip.tokenize(c) for c in self.list_concepts]
                        ).to(self.device)
                        self._cached_text_features = self.model_clip.encode_text(text_inputs)  # [num_concepts, D]

                    text_features = self._cached_text_features  # still on GPU

                    # Initialize similarity tensor for this batch
                    similarity = torch.zeros((images.size(0), len(self.list_concepts)), device=self.device)

                    # Process attentions (batch-level)
                    attentions, _ = self.prs.finalize(image_features)  # [B, L, N, H]
                    attentions = attentions[:,:,1:,:,:]
                    
                    if self.strategy in ['LTC', 'loc']:
                        B, L, N, H, E = attentions.shape
                        side = self.image_size // self.patch_size
                        
                        np.zeros((B, len(self.list_concepts), L, N, H))

                        # Process concepts
                        for i, text_feature in enumerate(text_features):
                            # Project attentions with concept vector
                            # Shape: [B, L, N, H, E]
                            proj = torch.einsum('blnhe,e->blnh', attentions, text_feature)

                            # Reshape into spatial maps [B*L*H, 1, side, side]
                            proj_maps = proj.reshape(B, L, side, side, H)
                            proj_maps = proj_maps.permute(0, 1, 4, 2, 3).reshape(-1, 1, side, side)

                            # Apply median filter on GPU
                            filtered = kornia.filters.median_blur(proj_maps, (3, 3))

                            # Restore shape [B, L, N, H]
                            filtered = filtered.reshape(B, L, H, side, side).permute(0, 1, 3, 4, 2)
                            filtered = filtered.reshape(B, L, N, H)

                            # Apply weights with broadcasting
                            weighted_attn = filtered * self.weights[None, :, None, :]
                            similarity[:, i] = weighted_attn.sum(dim=(1, 2, 3))

                        all_embeddings.append(similarity)

                    else:
                        image_features = self.model_clip.encode_image(images)
                        # Compute cosine similarity
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        similarity = (100.0 * image_features @ text_features.T)
                        all_embeddings.append(similarity)
                else:
                    image_features = self.model_clip.encode_image(images)
                    # Compute cosine similarity
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T)
                    all_embeddings.append(similarity)

        # If the file do not exists save the embeddings
        if self.disantangled_version is not False:
            if not os.path.exists(f'embeds/embeds_monumai_{self.disantangled_version}_{self.split}.npy'):
                np.save(f'embeds/embeds_monumai_{self.disantangled_version}_{self.split}.npy', np.array(torch.cat(all_embeddings, dim=0).cpu()))

        elif self.disantangled_version is False:
            if not os.path.exists(f'embeds/embeds_monumai_{self.split}.npy'):
                np.save(f'embeds/embeds_monumai_{self.split}.npy', np.array(torch.cat(all_embeddings, dim=0).cpu()))

        return torch.cat(all_embeddings, dim=0)
    
    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, idx):
        """Return precomputed image embeddings and label."""
        return {
            'clip_scores': self.image_embeddings[idx],
            'label_number': self.list_labels[idx]
        }