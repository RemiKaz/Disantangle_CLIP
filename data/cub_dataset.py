import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import clip
from tqdm import tqdm
import json
from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from clip_text_span.prs_hook import hook_prs_logger
import scipy

class CUBDataset(Dataset):
    """A dataset class to load and process images and annotations for the CUB dataset."""

    def __init__(self, root, phase, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', clip_concepts=None,
                 reduce_samples=False, device='cuda', select_segmentation='full', train_cbm_mode=False, class_restrict=None, disantangled_version=False):
        super(CUBDataset, self).__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.root = root
        self.phase = phase
        self.reduce_samples = reduce_samples
        self.model_name = model_name
        self.pretrained = pretrained
        self.select_segmentation = select_segmentation
        self.model_clip_name = model_name + pretrained
        self.train_cbm_mode = train_cbm_mode
        self.class_restrict = class_restrict
        self.clip_concepts = clip_concepts
        self.disantangled_version = disantangled_version

        # Load class labels and concepts from cub_per_class.json
        with open('/media/remi/RemiPro1/VLG-CBM/concept_files/cub_per_class.json', 'r') as f:
            cub_data = json.load(f)
            
        self.labels = list(cub_data.keys())
        self.dict_concepts = cub_data
        
        self.list_concepts = []
        for class_name in self.labels:
            self.list_concepts += self.dict_concepts[class_name]

        self.list_classes = self.labels

        # Load CLIP model and transforms based on provided model_name and pretrained options
        if 'laion' in pretrained:
            self.model_clip, _, self.transform = create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = get_tokenizer(model_name)
            self.model_clip.to(self.device)
            self.model_clip.eval()
        else:
            print('TODO make it work with OPENAI CLIP')
            exit()

        self.disantangled_version = disantangled_version
        if disantangled_version is not False:
            if disantangled_version == 'register':
                self.prs = hook_prs_logger(self.model_clip, self.device)
                self.strategy = 'register'
                self.image_size = 224 if model_name.startswith('ViT-B') else 256
                self.patch_size = 16 if model_name.startswith('ViT-B') else 14
            else:
                self.method_iou, self.strategy, self.type_concept = disantangled_version.split('-')
                self.prs = hook_prs_logger(self.model_clip, self.device)
                self.image_size = 224 if model_name.startswith('ViT-B') else 256
                self.patch_size = 16 if model_name.startswith('ViT-B') else 14
                self.weights = np.mean(np.load(f'weights/weights_cub_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'), axis=0)

        self.preprocess = self.transform
        self.load_data()

    def load_data(self):
        """Load and process images and annotations from the CUB dataset."""
        # Load the JSON file corresponding to the phase
        json_path = os.path.join(self.root, f'{self.phase}.json')
        with open(json_path, 'r') as f:
            self.paths_images = json.load(f)

        self.List_data_image = []
        for img_key, img_path in tqdm(self.paths_images.items(), desc="Loading CUB data"):
            # Load annotations (assuming img_key is used to fetch annotations)
            annotations = json.load(open(self.root+'/'+img_key))

            # For now, let's assume annotations are loaded from a file or generated elsewhere
            # Example: annotations = load_annotations_for(img_key)

            label = self.label_from_path(img_path)

            if self.class_restrict:
                for class_name in self.class_restrict:
                    
                    if label.lower().replace(' ','_') not in class_name.lower():
                        continue
                    
                    # Warning and pass is path do not exist
                    if os.path.exists(os.path.join(self.root, img_path)):
                        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
                        img_tensor = self.preprocess(img).to(self.device)
                    else:
                        print(f"Warning: Image {img_path} does not exist")
                        continue

                    # Load annotations and bounding boxes
                    annotations, bboxes = self.load_data_dataset_cub(annotations)
                    annotations = [ann.replace(' ','_') for ann in annotations]

                    # Translate/adapt labels if needed (adapt as per your dictionary)
                    # annotations = [self.Dict_spanish_english.get(ann, ann).replace('-', '_') for ann in annotations]

                    class_id = self.labels.index(label)

                    # Check if class name contains a hyphen
                    if '-' in class_name:
                        parts = class_name.split('-')
                        if parts[1] == 'random':
                            part_to_use = np.random.choice(annotations)
                            sub_concept = {'name_removed': part_to_use, 'segmentation': [], 'coco': None}
                        else:
                            sub_concept = {'name_removed': parts[1].replace(',',''), 'segmentation': [], 'coco': None}
                        # Update sub_concept segmentation if there's a match in annotations
                        for i, ann in enumerate(annotations):
                            if ann in sub_concept['name_removed']:
                                sub_concept['segmentation'].append(bboxes[i])
                    else:
                        # Handle full segmentation
                        sub_concept = {'name_removed': 'all', 'segmentation': bboxes, 'coco': None}

                    modified_image, mask = self.remove_sub_concept(img, sub_concept, return_mask=True)
                    img_tensor_only_concept = self.preprocess(modified_image).to(self.device)

                    clip_scores, image_features = self.compute_clip_scores(img_tensor, return_image_features=True)
                    clip_scores_only_concept, image_features_only_concept = self.compute_clip_scores(img_tensor_only_concept, return_image_features=True)

                    segmentation_data = [[i, bbox] for i, bbox in enumerate(bboxes)]
                    category_maps = {i: annotations[i] for i in range(len(annotations))}

                    if '-' in label:
                        self.List_data_image.append({
                            'concepts': annotations,
                            'label_number': class_id,
                            'class_name': label,
                            'image_path': img_path,
                            'image_tensor': img_tensor,
                            'image_tensor_only_concept': img_tensor_only_concept,
                            'image_features': image_features,
                            'image_features_only_concept': image_features_only_concept,
                            'segmentation_data': segmentation_data,
                            'concept_mask': sub_concept['name_removed'],
                            'mask': mask,
                            'clip_scores': clip_scores,
                            'clip_scores_only_concept': clip_scores_only_concept,
                            'json_class_pth': img_key,  # or the actual path if available
                            'catIds': category_maps
                        })
                    else:
                        self.List_data_image.append({
                            'concepts': annotations,
                            'label_number': class_id,
                            'class_name': label,
                            'image_path': img_path,
                            'image_tensor': img_tensor,
                            'segmentation_data': segmentation_data,
                            'concept_mask': sub_concept['name_removed'],
                            'clip_scores': clip_scores,
                            'json_class_pth': img_key,  # or the actual path if available
                            'catIds': category_maps
                        })
            else:
                
                # Warning and pass is path do not exist
                if os.path.exists(os.path.join(self.root, img_path)):
                    img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
                    img_tensor = self.preprocess(img).to(self.device)
                else:
                    print(f"Warning: Image {img_path} does not exist")
                    continue
                # Load annotations and bounding boxes
                annotations, bboxes = self.load_data_dataset_cub(annotations)

                # Process segmentation based on user selection
                sub_concept = self.process_segmentation(bboxes, annotations)
                
                # Generate mask from bounding boxes
                _, mask = self.remove_sub_concept(img, sub_concept, return_mask=True)

                class_id = self.labels.index(label)


                segmentation_data = [[i, bbox] for i, bbox in enumerate(bboxes)]
                '''plt.imshow(mask)
                plt.savefig('mask.png')
                plt.imshow(img)
                plt.savefig('image.png')
                exit()'''

                # Append data to List_data_image
                self.List_data_image.append({
                    'segmentation_data': segmentation_data,
                    'image_path': img_path,
                    'concepts': annotations,
                    'label_number': class_id,
                    'class_name': label,
                    'image_tensor': img_tensor,
                    'concept_mask': sub_concept['name_removed'],
                    'mask': mask,
                })

    def compute_clip_scores(self, image_tensor, return_image_features=False, no_class=None):
        """Computes CLIP scores for an image tensor."""
        text_inputs = torch.cat([clip.tokenize(concept) for concept in self.clip_concepts]).to(self.device)

        if self.disantangled_version:
            similarity = torch.zeros(len(self.clip_concepts), device=self.device)
            for i, text_input in enumerate(text_inputs):
                with torch.no_grad():
                    self.prs.reinit()
                    image_features = self.model_clip.encode_image(image_tensor.unsqueeze(0).to(self.device), attn_method="head", normalize=False)
                    text_features = self.model_clip.encode_text(text_input.unsqueeze(0).to(self.device)).squeeze().detach().cpu().numpy()
                    attentions, _ = self.prs.finalize(image_features)
                    attentions = attentions.detach().cpu().numpy()
                    attn_map = attentions[0, :, 1:] @ text_features

                    if self.strategy == 'LTC' or self.strategy == 'loc':
                        if self.weights is not None:
                            L, N, H = attn_map.shape
                            filtered_attn = np.zeros_like(attn_map)
                            for l in range(L):
                                for h in range(H):
                                    act = attn_map[l, :, h].reshape((self.image_size // self.patch_size, self.image_size // self.patch_size))
                                    filtered = scipy.ndimage.median_filter(act, size=3)
                                    filtered_attn[l, :, h] = filtered.reshape(-1)
                            extended_weights = np.tile(self.weights[:, np.newaxis, :], (1, N, 1))
                            weighted_attn = filtered_attn * extended_weights
                            similarity[i] = torch.tensor(weighted_attn).sum(axis=(0, 1, 2)).cpu()
                    elif self.strategy == 'register':
                        L, N, H = attn_map.shape
                        filtered_attn = np.zeros_like(attn_map)
                        for l in range(L):
                            for h in range(H):
                                act = attn_map[l, :, h].reshape((self.image_size // self.patch_size, self.image_size // self.patch_size))
                                filtered = scipy.ndimage.median_filter(act, size=3)
                                filtered_attn[l, :, h] = filtered.reshape(-1)
                        similarity[i] = torch.tensor(filtered_attn).sum(axis=(0, 1, 2)).cpu()

            clip_scores = {concept: similarity[i].item() for i, concept in enumerate(self.clip_concepts)}
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if return_image_features:
                return clip_scores, image_features
            return clip_scores

        with torch.no_grad():
            image_features = self.model_clip.encode_image(image_tensor.unsqueeze(0).to(self.device))
            text_features = self.model_clip.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * image_features @ text_features.T
            clip_scores = {concept: similarity[0][i].item() for i, concept in enumerate(self.clip_concepts)}
        if return_image_features:
            return clip_scores, image_features
        return clip_scores

    def process_segmentation(self, bboxes, annotations):
        """Process segmentation data based on user selection."""
        if self.select_segmentation == 'random':
            random_index = np.random.choice(len(annotations))
            name_removed = annotations[random_index]
            segmentation = [bboxes[random_index]]
        else:  # 'full'
            name_removed = 'all'
            segmentation = bboxes
        return {'name_removed': name_removed, 'segmentation': segmentation}

    def remove_sub_concept(self, image, sub_concept, inflation_radius=1, return_mask=False):
        """Create a mask from bounding box information."""
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        for bbox in sub_concept['segmentation']:
            xmin, ymin, xmax, ymax = bbox
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 255
        modified_image = image  # Placeholder for modified image logic
        if return_mask:
            return modified_image, mask
        return modified_image

    def label_from_path(self, path):
        """Extract label from image path."""
        for label in self.labels:
            if label.replace(' ', '_') in path:
                return label
        raise ValueError(f"No valid label found in path: {path}")

    def load_data_dataset_cub(self, annotations):
        """
        Extract attributes and bounding boxes from a list of annotation dictionaries.

        Args:
            annotations: List of dictionaries, each with 'label', 'logit', and 'box' keys.

        Returns:
            L_attributes: List of attribute labels.
            L_bboxes: List of bounding boxes as [xmin, ymin, xmax, ymax].
        """
        L_attributes = []
        L_bboxes = []

        for ann in annotations:
            if ann.get("label") is not None and ann.get("box") is not None:
                L_attributes.append(ann["label"])
                # Convert box coordinates to [xmin, ymin, xmax, ymax] format
                xmin, ymin, xmax, ymax = ann["box"]
                L_bboxes.append([xmin, ymin, xmax, ymax])

        return L_attributes, L_bboxes

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.List_data_image)

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset by index."""
        data = self.List_data_image[idx]
        if self.train_cbm_mode:
            return {
                'image_tensor': data['image_tensor'],
                'label_number': data['label_number']
            }
        return {
            'class_name': data['class_name'],
            'image_path': data['image_path'],
            'image_tensor': data['image_tensor'],
            'image_features': data.get('image_features'),
            'segmentation_data': data['segmentation_data'],
            'concepts': data['concepts'],
            'label_number': data['label_number'],
            'clip_scores': data.get('clip_scores'),
            'concept_mask': data['concept_mask'],
            'mask': data.get('mask')
        }