import numpy as np
import torch
from torch.nn import functional as F
import einops

def attn_map_to_heatmap(attn_map, class_embedding, model, device,model_name='ViT-B-32'):
    
    n_patches = int(attn_map.size()[2]-1)
    l_grid = int(np.sqrt(n_patches))
    
    '''attention_map = attn_map[0, :, 1:, :].sum(axis=(0, 2)) @ class_embedding.T'''
    
    attention_map = attn_map[0, :, 1:, :] @ class_embedding.T
    for h in range(attention_map.shape[0]):
        for l in range(attention_map.shape[2]):
            median_value = torch.max(attention_map[h, :, l])* 0.9
            for n in range(attention_map.shape[1]):
                if attention_map[h, n, l].max() > median_value:
                    # Ablate the neuron
                    attention_map[h, n, l] = 0
                    
    attention_map = attention_map.sum(axis=(0, 2)) 
    
    attention_map = F.interpolate(einops.rearrange(attention_map, '(B N M) C -> B C N M', N=l_grid, M=l_grid, B=1),
                                scale_factor=model.visual.patch_size[0],
                                mode='bilinear')[0].to(device).data

    Res = torch.clip(attention_map, 0, attention_map.max())
    # threshold between FG and BG is the mean
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    relevance = Res.permute(1, 2, 0).data.cpu().numpy()

    return relevance

def weight_neurons(attns,path_weights,remove_fst_layers=True,normalize=False):
    weights = np.expand_dims(np.load(path_weights),(0,2,4)) # [H,L,1] to [1,H,1,L,1]
    
    # Check if normalize is a tensor
    if isinstance(normalize, torch.Tensor):
        attns = attns - normalize
        
    weighted_attns = attns * weights
    '''if remove_fst_layers:
        weighted_attns[:,:10, :, :, :] = 0
    print(weighted_attns.sum(axis=(0, 2, 4)))
    print(weighted_attns.shape)'''
    return weighted_attns
    
def ablate_neuron(attns, head_id, layer_id):
    # Create a copy of the attention map
    ablated_attns = attns.copy()
    # Compute the mean value of the specified neuron
    '''replacement_value = ablated_attns[:, layer_id, head_id, :].mean()'''
    replacement_value = 0
    # Ablate the specified neuron with the mean value
    ablated_attns[:, layer_id,:, head_id, :] = replacement_value
    return ablated_attns

def ablate_neurons(attns, head_ids, layers_id):
    # Create a copy of the attention map
    ablated_attns = attns.copy()
    # Compute the mean value of the specified neurons
    for head_id, layer_id in zip(head_ids, layers_id):
        '''replacement_value = ablated_attns[:, layer_id, head_id, :].mean()'''
        replacement_value = 0
        # Ablate the specified neuron with the mean value
        ablated_attns[:, layer_id, head_id, :] = replacement_value
    return ablated_attns
