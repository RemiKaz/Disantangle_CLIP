import torch
import clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
from torch.nn import functional as F

def compute_clip_score_old(image_path, text_description, model_clip='ViT-B/32',pretrained=False):
    # Determine the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the appropriate CLIP model and preprocessor
    if 'laion' in pretrained:
        # Load the open-source version of CLIP
        from open_clip import create_model_and_transforms
        model, _, preprocess = create_model_and_transforms(model_clip, pretrained=pretrained)
    else:
        # Load the official CLIP model
        model, preprocess = clip.load(model_clip, device=device)

    model = model.to(device)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Save image preprocessed
    # revert image normalization
    
    # We reverse this normalization.
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

    # Reverse normalization: (x * std) + mean
    img_tensor = image_input.clone().detach().to(device)  # Move to CPU
    img_tensor = img_tensor * std[:, None, None]  # Un-normalize std
    img_tensor = img_tensor + mean[:, None, None]  # Un-normalize mean
    # Clamp to [0, 1] (in case of numerical errors)
    img_tensor = torch.clamp(img_tensor, 0, 1)

    # Convert to PIL Image
    img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

    # Save the image
    img_pil.save('save_path_image.png')

    # Tokenize the text description
    text_input = clip.tokenize([text_description]).to(device)

    # Compute CLIP features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Compute cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T)

    # Return the CLIP score
    return similarity.item()


def compute_clip_score(image,text_description,dataset):

    if type(image) == str:
        # Load and preprocess the image
        image = Image.open(image).convert("RGB")
        image_input = dataset.preprocess(image).unsqueeze(0).to(dataset.device)
        
    elif type(image) == torch.Tensor:
        image_input = image

    # Tokenize the text description
    text_input = clip.tokenize([text_description]).to(dataset.device)

    # Compute CLIP features
    with torch.no_grad():
        image_features = dataset.model.encode_image(image_input)
        text_features = dataset.model.encode_text(text_input)

    # Compute cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T)

    # Return the CLIP score
    return similarity.item()

def attn_map(image_tensor,text_description,dataset,prs,return_mlps=False):
    
    if 'laion' not in dataset.model_clip_name: # Check if open_clip variant
        ValueError('Model not supported')
        
    representation = dataset.model_clip.encode_image(image_tensor.unsqueeze(0).to(dataset.device), attn_method='head', normalize=False)

    attentions, mlps = prs.finalize(representation)
    
    texts = dataset.tokenizer(text_description).to(dataset.device)
    class_embeddings = dataset.model_clip.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)
    
    attention_map_1 = attentions[0, :, 1:, :].sum(axis=(1)) @ class_embedding.T
    
    '''print(attentions.size(),class_embedding.size())
    print(attention_map_1.size())
    attention_map_1 = F.interpolate(einops.rearrange(attentions, '(B N M) C -> B C N M', N=14, M=14, B=1),
                                 scale_factor=dataset.model.visual.patch_size[0],
                                 mode='bilinear').to(dataset.device)'''
    
    prs.reinit()
    
    if return_mlps:
        return attention_map_1[:,:,0],mlps
    
    return attention_map_1[:,:,0]

def save_heatmap_mosaic(attn_map_img_base, save_path):
    # Squeeze the tensor to remove the last dimension

    attn_map_img_base = attn_map_img_base.squeeze(-1)  # Shape: [1, H, N, L]

    attn_map_img_base = torch.clip(attn_map_img_base, torch.tensor(0).to(attn_map_img_base.device), attn_map_img_base.max())

    # Get the number of heads and lengths
    num_heads = attn_map_img_base.shape[0]
    num_patches = attn_map_img_base.shape[1]
    patch_length = attn_map_img_base.shape[2]

    # Calculate the number of rows and columns for the mosaic
    mosaic_rows = num_heads
    mosaic_cols = patch_length

    # Create a figure to hold the mosaic
    fig, axes = plt.subplots(mosaic_rows, mosaic_cols, figsize=(mosaic_cols * 2, mosaic_rows * 2))

    # Iterate over each head and length
    for h in range(num_heads):
        for l in range(patch_length):
            # Extract the attention map for the current head and length
            attn_map = attn_map_img_base[h, :, l]

            # Reshape the attention map to a 2D grid
            attn_map_2d = attn_map.reshape((int(np.sqrt(num_patches)), -1))

            # Normalize the attention map
            attn_map_2d = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() - attn_map_2d.min())

            # Plot the heatmap
            ax = axes[h, l]
            attn_map_2d_numpy = attn_map_2d.cpu().detach().numpy()
            ax.imshow(attn_map_2d_numpy, cmap='hot')
            ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the mosaic
    plt.savefig(save_path)
    plt.show()
    plt.clf()

def histogram_mosaic(attn_map_img_base, save_path):
    """
    Create a mosaic of histograms for each (head, layer) pair of the attention map,
    and print the max value on each subplot.

    Args:
        attn_map_img_base (torch.Tensor): Tensor of shape [H, N, L] or [1, H, N, L, 1]
        save_path (str): Path to save the histogram mosaic image.
    """
    # Handle case where input has extra dimensions
    if attn_map_img_base.dim() == 5:
        attn_map_img_base = attn_map_img_base.squeeze(0).squeeze(-1)  # [H, N, L]

    attn_map_img_base = torch.clip(
        attn_map_img_base, torch.tensor(0.0).to(attn_map_img_base.device),
        attn_map_img_base.max()
    )

    num_heads = attn_map_img_base.shape[0]
    attn_map_img_base.shape[1]
    num_layers = attn_map_img_base.shape[2]

    # Set up mosaic grid
    fig, axes = plt.subplots(num_heads, num_layers, figsize=(num_layers * 2.5, num_heads * 2.5))

    if num_heads == 1 and num_layers == 1:
        axes = np.array([[axes]])
    elif num_heads == 1 or num_layers == 1:
        axes = axes.reshape((num_heads, num_layers))

    for h in range(num_heads):
        for l in range(num_layers):
            values = attn_map_img_base[h, :, l].detach().cpu().numpy()

            ax = axes[h, l]
            n, bins, _ = ax.hist(values, bins=20, color='steelblue', alpha=0.85)
            ax.set_title(f'H{h} L{l}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            # Annotate max value
            max_val = np.max(values)*1000
            ax.text(0.95, 0.9, f"max: {max_val:.2f}", transform=ax.transAxes,
                    fontsize=6, ha='right', va='top', color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.clf()

def attn_map_bis(image_tensor,text_description,model_clip,model,tokenizer,device,prs,return_mlps=False,collapse='patches'):
    '''Same than attn_map but with other callables'''
    if 'laion' not in model_clip: # Check if open_clip variant
        ValueError('Model not supported')
   
    '''print(image_tensor.unsqueeze(0))'''

    representation = model.encode_image(image_tensor.unsqueeze(0).to(device), attn_method='head', normalize=False)
    
    attentions, mlps = prs.finalize(representation)
    texts = tokenizer(text_description).to(device)
    class_embeddings = model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)

    if collapse == 'patches':
        attentions[0,:,1:,:,:] @ class_embedding.T
        '''save_heatmap_mosaic(attention_map_1_bis,f"attention_map_{text_description}.png")
        histogram_mosaic(attention_map_1_bis,f"histogram_{text_description}.png")'''
        attention_map_1 = attentions[0,:,1:,:,:].sum(axis=(1)) @ class_embedding.T
        '''for h in range(attentions.shape[1]):
            for l in range(attentions.shape[3]):
                median_value = torch.max(attentions[:, h, :, l])* 0.5
                for n in range(attentions.shape[2]):
                    print(attentions.shape)
                    exit()
                    if attentions[0, h, n, l].max() > median_value:
                        # Ablate the neuron
                        attentions[:, h, n, l] = 0'''
        
    elif collapse == 'heads+layers':
        attention_map_1 = attentions[0,:,1:,:,:].sum(axis=(0,2)) @ class_embedding.T

    elif collapse == 'none':
        attention_map_1 = attentions[0,:,1:,:,:] @ class_embedding.T
        '''save_heatmap_mosaic(attention_map_1,'mosaic.png')'''
        
    prs.reinit()

    if return_mlps:
        if len(attention_map_1.shape) == 4:
            return attention_map_1[:,:,:,0],mlps
        
        return attention_map_1[:,:,0],mlps
    
    if len(attention_map_1.shape) == 4:
        return attention_map_1[:,:,:,0]
    
    return attention_map_1[:,:,0]

'''def plot_heatmap(attention_map, save_path, title):
    # Normalize the attention map
    v = attention_map
    min_ = attention_map.min()
    max_ = attention_map.max()
    v = v - min_
    v = np.uint8((v / (max_ - min_)) * 255)

    # Apply color map
    high = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    # Plot the heatmap
    plt.imshow(high)
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)'''
          
def compute_false_positive_rate(mask1, mask2):
    """
    Compute the false positive rate between two binary masks.

    Parameters:
    mask1 (numpy.ndarray): Predicted binary mask.
    mask2 (numpy.ndarray): Ground truth binary mask.

    Returns:
    float: The false positive rate.
    """
    # Ensure the masks are binary
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    # Calculate false positives: predicted as positive but actually negative
    false_positives = np.logical_and(mask1, np.logical_not(mask2))

    # Total predicted positives
    total_negatives = np.sum(np.logical_not(mask2))

    # Compute the false positive rate
    if total_negatives == 0:
        return 0.0  # Avoid division by zero
    false_positive_rate = np.sum(false_positives) / total_negatives

    return false_positive_rate
                
def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Parameters:
    mask1 (numpy.ndarray): First binary mask.
    mask2 (numpy.ndarray): Second binary mask.

    Returns:
    float: The IoU score.
    """
    
    # Ensure the masks are binary
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    # Calculate the intersection
    intersection = np.logical_and(mask1, mask2)

    # Calculate the union
    union = np.logical_or(mask1, mask2)

    # Compute the IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou