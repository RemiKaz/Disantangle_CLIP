import clip
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
import scipy
from clip_text_span.prs_hook import hook_prs_logger

class CLIPLinear(nn.Module):
    def __init__(self, list_concepts, list_classes, device, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',disantangled_version=False):
        super().__init__()
        """
        Initializes a CLIP_Linear instance.
        Args:
            list_concepts (list): List of concepts.
            list_classes (list): List of classes.
            device (str): Device to run on.
            model_name (str): Name of the model to use.
            pretrained (str): Pretrained weights identifier.
        """
        self.name = "CLIP-Linear"
        self.device = device
        self.disantangled_version = disantangled_version
        
        if 'laion' in pretrained:
            # Load the open-source version of CLIP
            self.clip_net, _, self.preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = get_tokenizer(model_name)
            self.clip_net.to(self.device)
            self.clip_net.eval()
        else:
            print('TODO make it work with OPENAI CLIP')
            exit()
            # Load the official CLIP model
            model = f"{model_name}/{pretrained}"
            self.clip_net, self.preprocess = clip.load(model, device=self.device)
            self.clip_net.eval()

        if disantangled_version is not False:
            self.method_iou, self.strategy, self.type_concept = disantangled_version.split('-') # median/treshold - LTC/loc - random/full
            # Load the disentangled model
            self.prs = hook_prs_logger(self.clip_net, self.device)
            self.image_size = 224 if model_name.startswith('ViT-B') else 256
            self.patch_size = 16 if model_name.startswith('ViT-B') else 14
            self.weights = np.mean(np.load(f'weights/weights_imagenet_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'),axis=0)  # Load weights for disentangled version

        # Store concepts and classes
        self.list_concepts = list_concepts
        tokens = clip.tokenize(list_concepts).to(self.device)
        self.text_embeding = self.clip_net.encode_text(tokens).float()
        self.list_classes = list_classes

        # Linear parameters
        self.linear = nn.Linear(len(list_concepts), len(list_classes)).to(self.device)

        # Custom colormap for matplotlib
        colors2 = plt.cm.coolwarm_r(np.linspace(0.5, 1, 128))
        colors1 = plt.cm.coolwarm_r(np.linspace(0, 0.5, 128))
        colors = np.vstack((colors2, colors1))
        mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
        mpl.cm.register_cmap("mycolormap", mymap)

    def forward(self, x):
        """Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        if self.disantangled_version and self.weights is not None:
            # Disentangled version with attention weighting
            text_inputs = torch.cat([clip.tokenize(concept) for concept in self.list_concepts]).to(self.device)

            # Initialize similarity tensor
            similarity = torch.zeros((x.size(0), len(self.list_concepts)), device=self.device)

            for batch_idx in range(x.size(0)):
                image_tensor = x[batch_idx].unsqueeze(0)  # Get single image from batch

                for i, text_input in enumerate(text_inputs):
                    with torch.no_grad():
                        if hasattr(self, 'prs'):  # Check if prs object exists
                            self.prs.reinit()
                            # Encode image with attention
                            image_features = self.clip_net.encode_image(
                                image_tensor.to(self.device),
                                attn_method="head",
                                normalize=False
                            )
                            # Encode text
                            text_feature = self.clip_net.encode_text(
                                text_input.unsqueeze(0).to(self.device)
                            ).squeeze().detach().cpu().numpy()

                            # Get attention maps
                            attentions, _ = self.prs.finalize(image_features)
                            attentions = attentions.detach().cpu().numpy()
                            attn_map = attentions[0, :, 1:] @ text_feature

                            # Apply filtering if using LTC or loc strategy
                            if self.strategy in ['LTC', 'loc']:
                                L, N, H = attn_map.shape
                                filtered_attn = np.zeros_like(attn_map)
                                for l in range(L):
                                    for h in range(H):
                                        act = attn_map[l, :, h].reshape((
                                            self.image_size // self.patch_size,
                                            self.image_size // self.patch_size
                                        ))
                                        filtered = scipy.ndimage.median_filter(act, size=3)
                                        filtered_attn[l, :, h] = filtered.reshape(-1)

                                # Apply weights
                                extended_weights = np.tile(
                                    self.weights[:, np.newaxis, :],
                                    (1, N, 1)
                                )
                                weighted_attn = filtered_attn * extended_weights
                                similarity[batch_idx, i] = torch.tensor(
                                    weighted_attn.sum(axis=(0, 1, 2))
                                ).cpu()
                        else:
                            # Fallback to standard CLIP if prs not available
                            image_features = self.clip_net.encode_image(image_tensor.to(self.device))
                            text_feature = self.clip_net.encode_text(text_input.unsqueeze(0).to(self.device))
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_feature /= text_feature.norm(dim=-1, keepdim=True)
                            similarity[batch_idx, i] = 100.0 * (image_features @ text_feature.T)

        else:
            # Standard CLIP forward pass
            with torch.no_grad():
                # Compute CLIP embeddings for input images
                clip_embeds = self.clip_net.encode_image(x)
                clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)

                # Encode text concepts
                text_inputs = torch.cat([clip.tokenize(concept) for concept in self.list_concepts]).to(self.device)
                text_features = self.clip_net.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity scores
                similarity = 100.0 * clip_embeds @ text_features.T

        # Pass through linear layer before returning
        return self.linear(similarity)
