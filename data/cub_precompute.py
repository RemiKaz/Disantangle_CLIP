import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import kornia
import clip
import json
import numpy as np
from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from clip_text_span.prs_hook import hook_prs_logger
import os
import random
from PIL import Image

class CUBPrecomputeDataset(Dataset):
    def __init__(
        self,
        root,
        split='train',
        model_name='ViT-B-16',
        pretrained='laion2b_s34b_b88k',
        device='cuda',
        pth_metadata='metadata/imagnet_classes.json',
        pth_metadata_parts='metadata/file_parts_mapping.json',
        list_concepts=None,
        restrict_samples=None,
        disantangled_version=False,
    ):
        """
        Args:
            root (str): Path to the ImageNet dataset.
            split (str): 'train', 'val', or 'test'.
            model_name (str): CLIP model name.
            pretrained (str): Pretrained weights for CLIP.
            device (str): Device to load tensors.
            pth_metadata (str): Path to metadata file for ImageNet classes.
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
        
        # Select 64 images for testing
        # Load class labels and concepts from cub_per_class.json
        with open('/media/remi/RemiPro1/disantangle_clip/concept_files/cub_per_class.json', 'r') as f:
            cub_data = json.load(f)
            
        self.list_classes = list(cub_data.keys())

        json_path = os.path.join(self.root, f'{self.split}.json')
        with open(json_path, 'r') as f:
            self.paths_images = json.load(f)

        # Reduce the number of samples for debugging
        if restrict_samples is not None:
            # This shuffling is not in-place
            l = list(self.paths_images.items())
            random.shuffle(l)
            self.paths_images = dict(l)

            self.paths_images = {k: v for k, v in self.paths_images.items() if k in list(self.paths_images.keys())[:restrict_samples]}
      
        self.list_labels = []
        
        for img_path in self.paths_images.values():
            self.list_labels.append(self.list_classes.index(self.label_from_path(img_path)))

        # Store concepts and classes
        self.disantangled_version = disantangled_version

        # Load class labels and concepts from cub_per_class.json
        with open('/media/remi/RemiPro1/disantangle_clip/concept_files/cub_per_class.json', 'r') as f:
            cub_data = json.load(f)
            
        self.list_classes = list(cub_data.keys())
        self.dict_concepts = cub_data
        
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
                np.load(f'weights/weights_cub_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'),
                axis=0
            )).to(self.device)

        # Precompute scores
        self.image_embeddings = self._precompute_image_embeddings()

    def label_from_path(self, path):
        """Extract label from image path."""
        for label in self.list_classes:
            if label.replace(' ', '_') in path:
                return label
        raise ValueError(f"No valid label found in path: {path}")

    def _precompute_image_embeddings(self, batch_size=32, num_workers=8):
        """Precompute image embeddings for the entire dataset with batching & GPU."""
        
        all_embeddings = []

        # load the embeddings if they exist
        if self.disantangled_version is not False:
            if os.path.exists(f'embeds/embeds_CUB_{self.disantangled_version}_{self.split}.npy'):
                all_embeddings = np.load(f'embeds/embeds_CUB_{self.disantangled_version}_{self.split}.npy')
                return torch.tensor(all_embeddings)

        elif self.disantangled_version is False:
            text_inputs = torch.cat(
                [clip.tokenize(c) for c in self.list_concepts]
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model_clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            if os.path.exists(f'embeds/embeds_CUB_{self.split}.npy'):
                all_embeddings = np.load(f'embeds/embeds_CUB_{self.split}.npy')
                return torch.tensor(all_embeddings)

        with torch.no_grad():
            for path_image in tqdm(self.paths_images.values(), desc="Precomputing image embeddings"):
                
                # Warning and pass is path do not exist
                if os.path.exists(os.path.join(self.root, path_image)):
                    # Batch image preprocessing
                    items = [Image.open(self.root+'/'+path_image).convert("RGB")]
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
            if not os.path.exists(f'embeds/embeds_CUB_{self.disantangled_version}_{self.split}.npy'):
                np.save(f'embeds/embeds_CUB_{self.disantangled_version}_{self.split}.npy', np.array(torch.cat(all_embeddings, dim=0).cpu()))

        elif self.disantangled_version is False:
            if not os.path.exists(f'embeds/embeds_CUB_{self.split}.npy'):
                np.save(f'embeds/embeds_CUB_{self.split}.npy', np.array(torch.cat(all_embeddings, dim=0).cpu()))

        return torch.cat(all_embeddings, dim=0)
    
    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, idx):
        """Return precomputed image embeddings and label."""
        return {
            'clip_scores': self.image_embeddings[idx],
            'label_number': self.list_labels[idx]
        }