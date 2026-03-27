import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import kornia
import clip
import json
import numpy as np
from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
import torchvision.datasets as datasets
from clip_text_span.prs_hook import hook_prs_logger
from torch.utils.data import DataLoader
import os

class ImageNetCLIPDataset(Dataset):
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
        part=None,
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
        self.part = part
        
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
        
        self.dataset = datasets.ImageNet(root=root, split=split, transform=self.preprocess)
        
        if restrict_samples is not None:
            print(len(self.dataset))
            print((self.part+1)*restrict_samples)
            indices = range(self.part*restrict_samples, (self.part+1)*restrict_samples)# restrict samples, position imposed by self.part        
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

        elif self.part is None:
            self.labels_dataset = torch.tensor(self.dataset.targets)

        elif disantangled_version is not None and self.split == 'train':
            indices = range(0, 1470000)# bad code, but the idea is the fusion of all samples is of this size 
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

        '''dataloader = DataLoader(self.dataset, shuffle=True)
        for i, batch in enumerate(dataloader):
            print(i, batch)
        exit()'''
        
        # Load metadata
        self.metadata_imagnet = json.load(open(pth_metadata))
        with open(pth_metadata_parts, 'r') as f:
            self.metadata_parts = json.load(f)

        self.list_concepts = []
        for class_name in self.metadata_parts.keys():
            self.list_concepts += self.metadata_parts[class_name]

        self.list_classes = self.metadata_imagnet.items()

        # Store concepts and classes
        self.disantangled_version = disantangled_version

        # Initialize disentangled version parameters
        if self.disantangled_version:
            self.method_iou, self.strategy, self.type_concept = disantangled_version.split('-')
            self.prs = hook_prs_logger(self.model_clip, self.device)
            self.image_size = 224 if model_name.startswith('ViT-B') else 256
            self.patch_size = 16 if model_name.startswith('ViT-B') else 14
            self.weights = torch.tensor(np.mean(
                np.load(f'weights/weights_imagenet_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'),
                axis=0
            )).to(self.device)

        # Precompute text embeddings
        if self.list_concepts is not None:
            tokens = clip.tokenize(self.list_concepts).to(self.device)
            with torch.no_grad():
                self.text_embeddings = self.model_clip.encode_text(tokens).float()

        # Precompute image embeddings
        self.image_embeddings = self._precompute_image_embeddings()

    def _precompute_image_embeddings(self, batch_size=16, num_workers=16):
        """Precompute image embeddings for the entire dataset with batching & GPU."""
        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=False)

        # load the embeddings if they exist

        if self.disantangled_version is not False and self.part is None:
            if os.path.exists(f'embeds/embeds_imagenet_{self.disantangled_version}_{self.split}_new.npy'):
                '''all_embeddings = np.memmap(
                        f'embeds/embeds_imagenet_{self.disantangled_version}_{self.split}_new.npy',
                        dtype=np.float32,
                        mode="r",  # read-only mode
                    )'''
                all_embeddings = np.load(f'embeds/embeds_imagenet_{self.disantangled_version}_{self.split}_new.npy')   
                return torch.tensor(all_embeddings)

        elif self.disantangled_version is False:
            text_inputs = torch.cat(
                [clip.tokenize(c) for c in self.list_concepts]
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model_clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            if os.path.exists(f'embeds/embeds_imagenet_{self.split}.npy'):
                all_embeddings = np.memmap(
                        f'embeds/embeds_imagenet_{self.split}.npy',
                        dtype=np.float32,
                        mode="r",  # read-only mode
                        shape=(len(self.dataset), len(self.list_concepts))
                    )
                return torch.tensor(all_embeddings)

        # Define embedding path depending on disentangled version
        if self.disantangled_version is not False:
            if self.part is not None:
                embed_path = f'embeds/embeds_imagenet_{self.disantangled_version}_{self.split}_{self.part}_new.npy'
            else:
                embed_path = f'embeds/embeds_imagenet_{self.disantangled_version}_{self.split}.npy'
        else:
            embed_path = f'embeds/embeds_imagenet_{self.split}.npy'

        # Create directory if needed
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)

        # Create memmap instead of torch.empty
        all_embeddings = np.memmap(
            embed_path,
            dtype=np.float32,
            mode="w+",  # create or overwrite
            shape=(len(self.dataset), len(self.list_concepts))
        )

        idx_batch = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Precomputing image embeddings"):
                items = batch[0]

                if isinstance(items, torch.Tensor):
                    images = items.to(self.device, non_blocking=True)
                else:
                    images = torch.stack([self.preprocess(img) for img in items]).to(self.device)

                if self.disantangled_version and self.weights is not None:
                    self.prs.reinit()

                    image_features = self.model_clip.encode_image(
                        images,
                        attn_method="head",
                        normalize=False
                    )

                    if not hasattr(self, "_cached_text_features"):
                        text_inputs = torch.cat([clip.tokenize(c) for c in self.list_concepts]).to(self.device)
                        self._cached_text_features = self.model_clip.encode_text(text_inputs)

                    text_features = self._cached_text_features
                    similarity = torch.zeros((images.size(0), len(self.list_concepts)), device=self.device)

                    attentions, _ = self.prs.finalize(image_features)  # [B, L, N, H]
                    attentions = attentions[:, :, 1:, :, :]

                    if self.strategy in ['LTC', 'loc']:
                        B, L, N, H, E = attentions.shape
                        side = self.image_size // self.patch_size

                        for i, text_feature in enumerate(text_features):
                            proj = torch.einsum('blnhe,e->blnh', attentions, text_feature)
                            proj_maps = proj.reshape(B, L, side, side, H)
                            proj_maps = proj_maps.permute(0, 1, 4, 2, 3).reshape(-1, 1, side, side)

                            filtered = kornia.filters.median_blur(proj_maps, (3, 3))
                            filtered = filtered.reshape(B, L, H, side, side).permute(0, 1, 3, 4, 2)
                            filtered = filtered.reshape(B, L, N, H)

                            weighted_attn = filtered * self.weights[None, :, None, :]
                            similarity[:, i] = weighted_attn.sum(dim=(1, 2, 3))

                    else:
                        image_features = self.model_clip.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        similarity = (100.0 * image_features @ text_features.T)

                    # Write directly into memmap
                    all_embeddings[idx_batch:idx_batch + images.size(0), :] = similarity.cpu().numpy()

                else:
                    image_features = self.model_clip.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T)
                    all_embeddings[idx_batch:idx_batch + images.size(0), :] = similarity.cpu().numpy()

                idx_batch += images.size(0)

        '''# Save all embeddings as an npy file
        np.save(f'embeds_{self.disantangled_version}_{self.split}_{self.part}.npy', np.array(torch.cat(all_embeddings, dim=0).cpu()))'''

        return all_embeddings
    
    def __len__(self):
        return len(self.labels_dataset)

    def __getitem__(self, idx):
        """Return precomputed image embeddings and label."""
        return {
            'clip_scores': self.image_embeddings[idx],
            'label_number': self.labels_dataset[idx]
        }