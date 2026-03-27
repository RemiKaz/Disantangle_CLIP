import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import os

from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from clip_text_span.utils.segmentation_utils import batch_pix_accuracy, batch_intersection_union, get_ap_scores
from clip_text_span.prs_hook import hook_prs_logger
import argparse
import shap
import pandas as pd
import data
import models
import clip
import json
import glob
import random

class SHAPCBM:
    """Class representing SHAP CBM explanations."""

    def __init__(self, model, dataset):
        """Initialize the SHAP CBM object.

        Parameters:
        ----------
        model: Concept Bottleneck Model to explain.
        """
        super().__init__()

        self.model = model

        self.model.to(self.model.device)

        '''x_train_summary = shap.kmeans(dataset.image_embeddings, 5)'''
        
        # Select 1000 image_embeddings randomly
        x_train_summary = dataset.image_embeddings[np.random.choice(len(dataset.image_embeddings), 1000, replace=False)].to(self.model.device)
    
        self.explainer = shap.DeepExplainer(self.model, x_train_summary
        )

    def compute_and_plot_explanation(self, scores, num_features=11, **kwargs):
        """Computes SHAP values for a given image and saves the explanation.

        Args:
            image (PIL.Image.Image): The input image.
            num_features (int): The number of features to plot.
            type_plot (str): The type of plot to use. Can be 'waterfall', 'force' or 'bar'.
                Defaults to 'waterfall'.
            kwargs (dict): Additional keyword arguments.
                - save_expl (str): The path to save the explanation image.
        """
        self.model.eval()
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        class_to_probe = kwargs["class_to_probe"] 
        img_path = kwargs["image_pth"]
        root = kwargs["root"]

        # Compute SHAP explanation for the input image
        d_data_in = {
            self.model.list_concepts[i]: scores[:, i].cpu().detach().numpy()
            for i in range(len(self.model.list_concepts))
        }
        pd_data_in = pd.DataFrame(d_data_in)

        shap_values = self.explainer.shap_values(scores)[0]

        if class_to_probe == 'prediction':
            # Infer the prediction from the model
            prediction = int(torch.argmax(self.model(scores[0])))

        elif class_to_probe == 'gt':

            def label_from_path(path):
                """Extract label from image path."""
                for label in self.model.list_classes:
                    if label.replace(' ', '_') in path:
                        return label
                raise ValueError(f"No valid label found in path: {path}")

            label = label_from_path(img_path)

            prediction = int(self.model.list_classes.index(label))
    
        if save_activations:
            shap_values_abs = np.abs(shap_values[:, prediction])
            attributes_copy = self.model.list_concepts.copy()
            _, shap_values_ordered, l_attributes_ordered = zip(
                *sorted(
                    zip(shap_values_abs, shap_values[:, prediction], attributes_copy, strict=False)
                ),
                strict=False,
            )

            d_data_in_all = {
                l_attributes_ordered[i]: shap_values_ordered[i]
                for i in range(len(shap_values_ordered))
            }
            print(d_data_in_all)

        self.model.train()

        # Plot the explanation
        # Waterfall variant

        shap.plots._waterfall.waterfall_legacy(
            self.explainer.expected_value[prediction],
            shap_values[:, prediction],
            features=scores[0],
            feature_names=self.model.list_concepts,
            max_display=num_features,
        )

        print(f"Saving SHAP plot to {save_expl}")
        plt.savefig(save_expl, bbox_inches="tight")
        plt.clf()
        plt.close()
        
        # Return activations

        shap_values_abs = np.abs(shap_values[:, prediction])
        attributes_copy = self.model.list_concepts.copy()
        _, shap_values_ordered, l_attributes_ordered = zip(
            *sorted(
                zip(shap_values_abs, shap_values[:, prediction], attributes_copy, strict=False)
            ),
            strict=False,
        )

        return self.model.list_classes[prediction],{
            l_attributes_ordered[i]: shap_values_ordered[i]
            for i in range(len(shap_values_ordered))
        }
        
def plot_exp_concept(concept, image_pth, weights_pth, device='cuda:0', model_name='ViT-B-16', pretrained_name='laion2b_s34b_b88k',
                     root=None):
    # --- Load model ---
    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained_name)
    model.to(device).eval()
    tokenizer = get_tokenizer(model_name)
    prs = hook_prs_logger(model, device)
    print(root)
    # --- Load image ---
    image = preprocess(Image.open(image_pth).convert('RGB')).unsqueeze(0).to(device)

    # --- Classifier & weights ---
    weights = np.load(weights_pth)

    # --- Encode and extract attention ---
    prs.reinit()
    representation = model.encode_image(image, attn_method="head", normalize=False)
    attentions, _ = prs.finalize(representation)
    attentions = attentions.detach().cpu().numpy()  # [b, l, n, h, d]

    text_embedding = model.encode_text(tokenizer(concept).to(device)).detach().cpu().numpy()

    attn_map = attentions[0, :, 1:] @ text_embedding[0]

    # --- Aggregate & interpolate ---
    Res0_torch = torch.tensor(attn_map).sum(axis=(0, 2)).reshape(image_size // patch_size, image_size // patch_size).unsqueeze(0).unsqueeze(0).cpu()
    Res0 = torch.nn.functional.interpolate(Res0_torch, scale_factor=patch_size, mode='bilinear').squeeze().cpu()
    Res0 = torch.clip(Res0, 0, Res0_torch.max())
    Res0 = (Res0 - Res0.min()) / (Res0.max() - Res0.min())
    # --- Apply median filter ---
    L, N, H = attn_map.shape
    filtered_attn = np.zeros_like(attn_map)
    for l in range(L):
        for h in range(H):
            act = attn_map[l, :, h].reshape((image_size // patch_size, image_size // patch_size))
            filtered = scipy.ndimage.median_filter(act, size=3)
            filtered_attn[l, :, h] = filtered.reshape(-1)

    # --- Aggregate & interpolate ---
    Res05_torch = torch.tensor(filtered_attn).sum(axis=(0, 2)).reshape(image_size // patch_size, image_size // patch_size).unsqueeze(0).unsqueeze(0).cpu()
    Res05 = Res0_torch - Res05_torch
    Res05 = torch.nn.functional.interpolate(Res05, scale_factor=patch_size, mode='bilinear').squeeze().cpu()
    Res05 = torch.clip(Res05, 0, Res05.max())
    Res05 = (Res05 - Res05.min()) / (Res05.max() - Res05.min())

    # --- Apply weights ---

    _, N, _ = filtered_attn.shape

    extended_weights = np.tile(np.mean(weights,axis=0)[:, np.newaxis, :], (1, N, 1))

    extended_weights = extended_weights > np.max(extended_weights) *0.9

    weighted_attn = filtered_attn * extended_weights

    # --- Aggregate & interpolate ---
    Res055_torch = torch.tensor(weighted_attn).sum(axis=(0, 2)).reshape(image_size // patch_size, image_size // patch_size).unsqueeze(0).unsqueeze(0).cpu()
    Res055 = Res05_torch - Res055_torch
    Res055 = torch.nn.functional.interpolate(Res055, scale_factor=patch_size, mode='bilinear').squeeze().cpu()
    Res055 = torch.clip(Res055, 0, Res055.max())
    Res055 = (Res055 - Res055.min()) / (Res055.max() - Res055.min())

    Res2 = Res0_torch - Res055_torch
    Res2 = torch.clip(Res2, 0, Res2.max())
    Res2 = (Res2 - Res2.min()) / (Res2.max() - Res2.min())

    Res055_torch = torch.nn.functional.interpolate(Res055_torch, scale_factor=patch_size, mode='bilinear').squeeze().cpu()

    # --- Normalize and threshold ---

    power_object = Res055_torch.sum()/(Res055_torch.sum()+Res055.sum()+Res05.sum())

    Res = torch.clip(Res055_torch, 0, Res055_torch.max())

    Res = (Res - Res.min()) / (Res.max() - Res.min())*power_object
    thr = Res.mean()
    Res_1 = Res > thr
    Res_0 = Res <= thr

    # --- Plotting ---
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    power_context = (Res055.sum()+Res05.sum())/(Res055_torch.sum()+Res055.sum()+Res05.sum())

    # Merge register and context part
    Res_context = (0.5 * Res05 + 0.5 * Res055)*power_context

    # Original image
    axs[0].imshow(Image.open(image_pth).resize((image_size, image_size)))
    axs[0].set_title('Original Image')

    # Save image resised
    
    Image.open(image_pth).resize((image_size, image_size)).save(f'{root}/input_image.png')

    # Assuming 'image' is the original image (numpy array normalized to [0,1])
    # and it's available in this scope

    # Original image
    axs[0].imshow(Image.open(image_pth).resize((image_size, image_size)))
    axs[0].set_title('Original Image')

    # Superimpose each heatmap on the image with 50% transparency
    axs[1].imshow(Image.open(image_pth).resize((image_size, image_size)))
    axs[1].imshow(Res0.numpy(), cmap='jet', alpha=0.5)  # alpha=0.5 for 50% transparency
    axs[1].set_title('Original heatmap')

    axs[2].imshow(Image.open(image_pth).resize((image_size, image_size)))
    axs[2].imshow(Res_context.numpy(), cmap='jet', alpha=0.5)
    axs[2].set_title('Context part')
    
    axs[3].imshow(Image.open(image_pth).resize((image_size, image_size)))
    axs[3].imshow(Res.numpy(), cmap='jet', alpha=0.5)
    axs[3].set_title('Content part')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{root}/{concept}_map.png')
    
    plt.show()
    plt.clf()
    plt.close(fig)

    return Res0.numpy(), Res05.numpy(), Res.numpy()
    

if __name__ == '__main__':
    # Training settings
    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--dataset", type=str, default="cub", help="The name of the dataset")
        parser.add_argument("--dataset_root", type=str, default="ROOT_DATASET/CUB", help="Root path to the dataset")
        parser.add_argument("--model_pth", type=str, default="best_model_cub_precompute_clip_linear_precompute_median-loc-random.pt", help="Path to the trained CBM model")
        parser.add_argument("--weights_pth", type=str, default="weights/weights_cub_median_random_loc_all_classes.npy", help="Path to the attention weights")
        parser.add_argument("--image_paths", type=str, nargs='+', default=['ROOT_DATASET/CUB/test/189.Red_bellied_Woodpecker/Red_Bellied_Woodpecker_0031_180975.jpg'], help="List of image paths to explain")
        parser.add_argument("--output_dir", type=str, default="./one_sample_output", help="Directory to save results")
        parser.add_argument("--model_name", type=str, default="ViT-B-16", help="CLIP model name")
        parser.add_argument("--pretrained_name", type=str, default="laion2b_s34b_b88k", help="CLIP pretrained weights")
        parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
        parser.add_argument("--disantangled_version", type=str, default="median-loc-random", help="Disentangled version to use")
        parser.add_argument("--restrict_samples", type=int, default=None, help="Number of samples to restrict to")
        return parser.parse_args()

    # Load data/explainer/model
    args = parse_args()

    model_name = args.model_name
    pretrained_name = args.pretrained_name
    weights_pth = args.weights_pth
    device = args.device
    restrict_samples = args.restrict_samples
    disantangled_version = args.disantangled_version

    if model_name.startswith('ViT-B'):
        image_size = 224
        patch_size = 16
    elif model_name.startswith('ViT-H'):
        image_size = 256
        patch_size = 16
    elif model_name.startswith('ViT-L'):
        image_size = 224
        patch_size = 14
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_clip, _, transform = create_model_and_transforms(model_name, pretrained=pretrained_name)
    tokenizer = get_tokenizer(model_name)
    model_clip.to(device)
    model_clip.eval()

    # Load concepts and classes
    if args.dataset == 'cub':

        # Load dataset
        dataset_train = data.CUBPrecomputeDataset(args.dataset_root,
                                                split='val', model_name=model_name, pretrained=pretrained_name,
                            device='cuda',disantangled_version=disantangled_version,restrict_samples=restrict_samples)

        list_concepts = dataset_train.list_concepts
        list_classes = dataset_train.list_classes

    # Load model
    model = models.CLIPLinearPrecomputed(list_concepts, list_classes, device=device)
    model.load_state_dict(torch.load(args.model_pth))

    # Compute CLIP scores
    text_inputs = torch.cat(
            [clip.tokenize(c) for c in list_concepts]
        ).to(device)

    with torch.no_grad():
        text_features = model_clip.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    selected_paths = args.image_paths

    for image_pth in selected_paths :
        
        name_dir_result = image_pth.split('/')[-1].split('.')[0]
        os.makedirs(os.path.join(output_dir, name_dir_result), exist_ok=True)
        dir_sample = os.path.join(output_dir, name_dir_result)
        
        image_tensor = dataset_train.preprocess(Image.open(image_pth).convert('RGB')).unsqueeze(0).to(device)
            
        with torch.no_grad():
            image_features = model_clip.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            scores = (100.0 * image_features @ text_features.T)

        explainer = SHAPCBM(model, dataset_train)
        
        class_prediction, dict_activations = explainer.compute_and_plot_explanation(scores, num_features=6, save_expl=os.path.join(dir_sample, f'test_SHAP.png'),
                                           save_activations=None, return_activations=False, class_to_probe='gt', image_pth=image_pth,
                                           root=dir_sample)
        
        top_3_concepts = list(dict_activations.keys())[-5:]
        
        activations_dict = {'path_img': image_pth, 'prediction': class_prediction, 'concepts': {}}
        for concept_text in top_3_concepts:
            activations_dict['concepts'][concept_text] = {}
            _ ,_ , activations = plot_exp_concept(concept_text, image_pth, weights_pth, device=device, model_name=model_name, 
                                                  pretrained_name=pretrained_name,root=dir_sample)
            
            activations_dict['concepts'][concept_text]['activations'] = activations.tolist()
            activations_dict['concepts'][concept_text]['shap_value'] = dict_activations[concept_text]
            
        # Save activations
        json.dump(activations_dict, open(os.path.join(dir_sample, f'activations.json'), 'w'))
    