import glob
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import clip
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from clip_text_span.prs_hook import hook_prs_logger
import scipy

class MonumaiDataset(Dataset):
    """A dataset class to load and process images and annotations for the Monumai dataset."""

    def __init__(self, root, phase, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',clip_concepts=None,
                 reduce_samples=False, device='cuda', select_segmentation='full', train_cbm_mode=False, class_restrict=None,disantangled_version=False):
        super(MonumaiDataset, self).__init__()
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

        # Define labels and attributes
        self.labels = ['Baroque', 'Gothic', 'Hispanic-Muslim', 'Renaissance']
        self.Dict_spanish_english = {
            'arco-herradura': 'horseshoe-arch', 'arco-lobulado': 'lobed-arch',
            'arco-apuntado': 'pointed-arch', 'arco-conopial': 'ogee-arch',
            'arco-trilobulado': 'trefoil-arch', 'serliana': 'serliana',
            'columna-salomonica': 'solomonic-column', 'pinaculo-gotico': 'pinnacle-gothic',
            'ojo-de-buey': 'porthole', 'fronton-partido': 'broken-pediment',
            'arco-medio-punto': 'rounded-arch', 'dintel-adovelado': 'flat-arch',
            'fronton-curvo': 'segmental-pediment', 'fronton': 'triangular-pediment',
            'vano-adintelado': 'lintelled-doorway'
        }

        self.Dict_concepts = {
            'Hispanic-muslim': ['Horseshoe arch', 'Lobed arch', 'Flat arch'],
            'Gothic': ['Pointed arch', 'Ogee arch', 'Trefoil arch', 'Gothic pinnacle'],
            'Renaissance': ['Triangular pediment', 'Segmental pediment', 'Serliana', 'Porthole', 'Lintelled doorway', 'Rounded arch'],
            'Baroque': ['Porthole', 'Lintelled doorway', 'Rounded arch', 'Broken pediment', 'Solomonic column']
        }

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
                self.method_iou, self.strategy, self.type_concept = disantangled_version.split('-') # median/treshold - LTC/loc - random/full
                # Load the disentangled model
                self.prs = hook_prs_logger(self.model_clip, self.device)
                self.image_size = 224 if model_name.startswith('ViT-B') else 256
                self.patch_size = 16 if model_name.startswith('ViT-B') else 14
                print('Warning, weights of imagenet have been put for ablation')
                self.weights = np.mean(np.load(f'weights/weights_imagenet_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'),axis=0)  # Load weights for disentangled version
                
        self.preprocess = self.transform
        self.load_data()

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

    def load_data(self):
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

        if self.phase == 'train':
            self.selected_img_paths, self.selected_annotations_paths = img_paths_train, ann_paths_train
        elif self.phase == 'val':
            self.selected_img_paths, self.selected_annotations_paths = img_paths_val, ann_paths_val
        elif self.phase == 'test':
            self.selected_img_paths, self.selected_annotations_paths = img_paths_test, ann_paths_test
        else:
            raise ValueError("Invalid phase specified. Choose from 'train', 'val', or 'test'.")

        if self.reduce_samples:
            slice_length = 50
            self.selected_img_paths = self.selected_img_paths[:slice_length]
            self.selected_annotations_paths = self.selected_annotations_paths[:slice_length]

        for img_path, xml_path in tqdm(zip(self.selected_img_paths, self.selected_annotations_paths), total=len(self.selected_img_paths)):

            if self.class_restrict:
                
                label = self.label_from_path(img_path)
                for class_name in self.class_restrict:
                    
                    if label.lower() not in class_name:
                        continue
                    
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.preprocess(img).to(self.device)

                    annotations, bboxes = self.load_data_dataset_monumai(xml_path)
                    
                    # Translate/adapt labels
                    annotations = [self.Dict_spanish_english[ann].replace('-','_') for ann in annotations]
                    
                    class_id = self.labels.index(label)    
                
                    # Check if class name contains a hyphen
                    if '-' in class_name:
                        parts = class_name.split('-')
                        if parts[1] == 'random':
                            part_to_use = np.random.choice(annotations)
                            sub_concept = {'name_removed': part_to_use, 'segmentation': [], 'coco': None}
                        else:
                            sub_concept = {'name_removed': parts[1], 'segmentation': [], 'coco': None}

                        # Update sub_concept segmentation if there's a match in annotations
                        for i, ann in enumerate(annotations):
                            if ann == sub_concept['name_removed']:
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
                            'json_class_pth': xml_path,
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
                            'json_class_pth': xml_path,
                            'catIds': category_maps
                        })
                    
            else:
                
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img).to(self.device)

                annotations, bboxes = self.load_data_dataset_monumai(xml_path)
                label = self.label_from_path(img_path)
                class_id = self.labels.index(label)
                # Load image
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img).to(self.device)
                # Parse XML to get annotations and bounding boxes
                annotations, bboxes = self.load_data_dataset_monumai(xml_path)
                # Determine the class label and id
                # Process segmentation based on user selection
                sub_concept = self.process_segmentation(bboxes, annotations)
                # Generate mask from bounding boxes
                _, mask = self.remove_sub_concept(img, sub_concept, return_mask=True)
                # Plot and save the mask and image
                '''
                plt.imshow(mask)
                plt.savefig('mask.png')
                plt.imshow(img)
                plt.savefig('image.png')
                exit()
                '''
                # Append data to List_data_image
                self.List_data_image.append({
                    'concepts': annotations,
                    'label_number': class_id,
                    'class_name': label,
                    'image_tensor': img_tensor,
                    'concept_mask': sub_concept['name_removed'],
                    'mask': mask,
                })

    def get_cat_ids(self, annotations):
        """Mock method to simulate the catIds mapping."""
        return {i: name for i, name in enumerate(annotations)}

    def compute_clip_scores(self, image_tensor, return_image_features=False, no_class=None):
        """Computes CLIP scores for an image tensor."""
        
        text_inputs = torch.cat([clip.tokenize(concept) for concept in self.clip_concepts]).to(self.device)
        
        if self.disantangled_version:
            similarity = torch.zeros(len(self.clip_concepts), device=self.device)
            for i,text_input in enumerate(text_inputs):
                with torch.no_grad():
                    self.prs.reinit()
                    image_features = self.model_clip.encode_image(image_tensor.unsqueeze(0).to(self.device), attn_method="head", normalize=False)
                    text_features = self.model_clip.encode_text(text_input.unsqueeze(0).to(self.device)).squeeze().detach().cpu().numpy()
                    attentions, _ = self.prs.finalize(image_features)
                    attentions = attentions.detach().cpu().numpy()
                    attn_map = attentions[0, :, 1:] @ text_features
                              
                    # New version
                    
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
                            #extended_weights = np.tile(self.weights[no_class][:, np.newaxis, :], (1, N, 1))
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
            
            else:        
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
            mask[ymin:ymax, xmin:xmax] = 255
        modified_image = image  # Placeholder for modified image logic
        if return_mask:
            return modified_image, mask
        return modified_image

    def label_from_path(self, path):
        """Extract label from image path."""
        for label in self.labels:
            if label in path:
                return label
        raise ValueError(f"No valid label found in path: {path}")

    def load_data_dataset_monumai(self, xml_path):
        """Extract attributes and bounding boxes from XML annotation file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        L_attributes = []
        L_bboxes = []
        for obj in root.findall('object'):
            bbox = [
                int(obj.find('bndbox/xmin').text), int(obj.find('bndbox/ymin').text),
                int(obj.find('bndbox/xmax').text), int(obj.find('bndbox/ymax').text)
            ]
            L_bboxes.append(bbox)
            name = obj.find('name').text
            L_attributes.append(name)
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

        if self.class_restrict and '-' in data['class_name']:
            class_parts = data['class_name'].split('-')
            if class_parts[1] == 'random':
                part_to_use = np.random.choice(data['concepts'])
                data['concept_mask'] = part_to_use
            else:
                data['concept_mask'] = class_parts[1]

        return {
            'class_name': data['class_name'],
            'image_path': data['image_path'],
            'image_tensor': data['image_tensor'],
            'image_tensor_only_concept': data.get('image_tensor_only_concept'),
            'image_features': data.get('image_features'),
            'image_features_only_concept': data.get('image_features_only_concept'),
            'segmentation_data': data['segmentation_data'],
            'concepts': data['concepts'],
            'label_number': data['label_number'],
            'catIds': data['catIds'],
            'clip_scores': data.get('clip_scores'),
            'clip_scores_only_concept': data.get('clip_scores_only_concept'),
            'json_class_pth': data['json_class_pth'],
            'concept_mask': data['concept_mask'],
            'mask': data.get('mask')
        }
