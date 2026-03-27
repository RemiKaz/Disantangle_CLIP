import json
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
import torch
import clip
import numpy as np
import cv2
from clip_text_span.utils.factory import get_tokenizer
from tqdm import tqdm
import utils
import scipy
from clip_text_span.prs_hook import hook_prs_logger

class PIN_dataset(Dataset):
    def __init__(self, root_data,root_parts,split,pth_metadata='metadata/imagnet_classes.json', pth_metadata_parts='metadata/file_parts_mapping.json',
                 class_restrict=None,select_segmentation='full', clip_concepts=None,model_name='ViT-B-16',pretrained='laion2b_s34b_b88k',
                 disantangled_version=False,reduce_samples=False,train_cbm_mode=False,alpha=10):
        # Load metadata
        self.metadata_imagnet = json.load(open(pth_metadata))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the list to store image data
        self.List_data_image = []
        self.model_clip_name = model_name + pretrained
        # Initialize CLIP model and processor if clip_concepts is provided
        self.clip_concepts = clip_concepts
        self.train_cbm_mode = train_cbm_mode
        
        # Load the appropriate CLIP model and preprocessor
        if 'laion' in pretrained:
            # Load the open-source version of CLIP
            from clip_text_span.utils.factory import create_model_and_transforms
            self.model_clip, _, self.transform = create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = get_tokenizer(model_name)
            self.model_clip.to(self.device)
            self.model_clip.eval()
        else:    
            print('TODO make it work with OPENAI CLIP')
            exit()
            # Load the official CLIP model
            self.model_clip, self.transform = clip.load(model, device=self.device)
            self.model_clip.eval()

        self.flag_import_in_getitem = False
        self.disantangled_version = disantangled_version
        
        if disantangled_version is not False:
            self.method_iou, self.strategy, self.type_concept = disantangled_version.split('-') # median/treshold - LTC/loc - random/full
            # Load the disentangled model
            self.prs = hook_prs_logger(self.model_clip, self.device)
            self.image_size = 224 if model_name.startswith('ViT-B') else 256
            self.patch_size = 16 if model_name.startswith('ViT-B') else 14
            if alpha == 10:
                print('Warning, weights of monumai have been put for ablation')
                self.weights = np.mean(np.load(f'weights/weights_monumai_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'),axis=0)  # Load weights for disentangled version
            else:
                self.weights = np.mean(np.load(f'weights/weights_imagenet_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes_{alpha}.npy'),axis=0)  # Load weights for disentangled version
            
        with open(pth_metadata_parts, 'r') as f:
            self.metadata_parts = json.load(f)
            
        # If class_restrict is provided, filter classes
        if class_restrict:
            for class_name in class_restrict:
                # Get the class ID from metadata
                no_class = self.metadata_imagnet[class_name.split('-')[0]]['cls_id']
                id_class = self.metadata_imagnet[class_name.split('-')[0]]['id']

                # Construct the path to the JSON file for this class
                json_class_pth = f"{root_parts}/json/{no_class}.json"

                # Load the COCO dataset for this class
                coco = COCO(json_class_pth)
                catIds = coco.getCatIds(catNms=[class_name.split('-')[0]])

                # Iterate over each image in the COCO dataset
                for image_id in coco.getImgIds(catIds=catIds):
                    # Get image info
                    image_info = coco.loadImgs(image_id)[0]
                    image_path = f"{root_data}/train/{image_info['file_name']}"

                    if '-' in class_name:
                        parts = class_name.split('-')
                        
                        if parts[1] == 'random': # Select a random part in the class of the image and save it if present
                            part_to_use = np.random.choice(self.metadata_parts[no_class])
                            sub_concept = {'name_removed':part_to_use,'segmentation':[],'coco':coco}
                            
                        else:
                            sub_concept = {'name_removed':parts[1],'segmentation':[],'coco':coco}
                            
                        # Get annotations (segmentation map)
                        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
                        anns = coco.loadAnns(ann_ids)

                        segmentation_data = []
                        category_maps = {}
                        for ann in anns:
                            category_name = coco.loadCats(int(ann['category_id']))[0]["name"].replace(' ', '_')
                            if category_name == sub_concept['name_removed']:
                                sub_concept['segmentation'].append(ann['segmentation'])

                    # Get annotations (segmentation map)
                    ann_ids = coco.getAnnIds(imgIds=image_info['id'])
                    anns = coco.loadAnns(ann_ids)
                    segmentation_data = [[ann['category_id'], ann['segmentation']] for ann in anns]
                    category_maps = {ann['category_id']: coco.loadCats(int(ann['category_id']))[0]["name"] for ann in anns}
                    concepts = [coco.loadCats(int(ann['category_id']))[0]["name"].replace(' ','_') for ann in anns]

                    # Load image as tensor
                    image_tensor = self.load_image_as_tensor(image_path)
                    
                    if '-' in class_name:
                        image_tensor_only_concept, mask = self.load_image_as_tensor(image_path,impait_part=sub_concept,return_mask=True)
                        clip_scores_only_concept, image_features_only_concept = self.compute_clip_scores(image_tensor_only_concept, return_image_features=True,no_class=id_class) if self.clip_concepts else None
                    
                    # Compute CLIP scores if clip_concepts is provided
                    clip_scores, image_features = self.compute_clip_scores(image_tensor, return_image_features=True,no_class=id_class) if self.clip_concepts else None

                    if '-' in class_name:
                        # Append the data to the list
                        self.List_data_image.append({
                            'class_name': class_name,
                            'image_path': image_path,
                            'image_tensor': image_tensor,
                            'image_tensor_only_concept': image_tensor_only_concept,
                            'image_features': image_features,
                            'image_features_only_concept': image_features_only_concept,
                            'segmentation_data': segmentation_data,
                            'concepts': concepts,
                            'label_number': no_class,
                            'catIds': category_maps,
                            'clip_scores': clip_scores,
                            'clip_scores_only_concept': clip_scores_only_concept,
                            'json_class_pth': json_class_pth,
                            'concept_mask': sub_concept['name_removed'],
                            'mask': mask
                        })

                    else:
                        # Append the data to the list
                        self.List_data_image.append({
                            'class_name': class_name,
                            'image_path': image_path,
                            'image_tensor': image_tensor,
                            'segmentation_data': segmentation_data,
                            'concepts': concepts,
                            'label_number': no_class,
                            'catIds': category_maps,
                            'clip_scores': clip_scores,
                            'concept_mask': sub_concept['name_removed'],
                        })
                        
        else : # Version with all classes, do not compute CLIP scores, and return masks associated to the entire class
            
            # For debug
            '''def slice_dict(d, start=0, end=None):
                """Returns a slice of the dictionary from index `start` to `end`."""
                items = list(d.items())
                sliced_items = items[start:end]
                return dict(sliced_items)
                        
            self.metadata_imagnet = slice_dict(self.metadata_imagnet, start=0, end=3)'''
            
            if reduce_samples:
                print("Reducing samples for faster testing/debugging.")
            
            # Iterate over all classes in the metadata
            for class_name, class_info in tqdm(self.metadata_imagnet.items()):
                no_class = class_info['cls_id']

                # Construct the path to the JSON file for this class
                json_class_pth = f"{root_parts}/json/{no_class}.json"

                # Load the COCO dataset for this class
                with utils.HiddenPrints():
                    coco = COCO(json_class_pth)
                catIds = coco.getCatIds(catNms=[class_name])

                if reduce_samples:
                    slice = 2
                
                else:
                    slice = None

                # Iterate over each image in the COCO dataset
                for image_id in coco.getImgIds(catIds=catIds)[:slice]:
                    # Get image info
                    image_info = coco.loadImgs(image_id)[0]
                    image_path = f"{root_data}/train/{image_info['file_name']}"

                    # Get annotations (segmentation map)
                    anns = coco.loadAnns(coco.getAnnIds(imgIds=image_info['id']))
                    concepts = [coco.loadCats(int(ann['category_id']))[0]["name"].replace(' ', '_') for ann in anns]

                    if select_segmentation == 'random': # Select a random part in the class of the image and save it if present
                        part_to_use = np.random.choice(self.metadata_parts[no_class])
                        sub_concept = {'name_removed':part_to_use,'segmentation':[],'coco':coco}
                    
                        category_maps = {}
                        for ann in anns:
                            category_name = coco.loadCats(int(ann['category_id']))[0]["name"].replace(' ', '_')
                            if category_name == sub_concept['name_removed']:
                                sub_concept['segmentation'].append(ann['segmentation'])
                        
                    elif select_segmentation == 'full': # Handle the case where concept is 'all'
                        sub_concept = {'name_removed': 'all', 'segmentation': [ann['segmentation'] for ann in anns], 'coco': coco}
                        
                    _, mask = self.load_image_as_tensor(image_path, impait_part=sub_concept, return_mask=True)

                    # Append only the necessary data to the list
                    self.List_data_image.append({
                        'concepts': concepts,
                        'label_number': no_class,
                        'image_tensor': image_path,
                        'concept_mask': sub_concept['name_removed'],
                        'mask': mask,
                    })
            
            self.flag_import_in_getitem = True

    def save_image_id(self, idx, wanted_concept=None,save_also_only_concept=False,save_also_mask=False):
        # Save the image at the specified index to the given path
        
        actual_id = 0
        
        if wanted_concept :
            for data_sample in self.List_data_image:
                if wanted_concept in data_sample['concepts']:     
                    if actual_id == idx :  
                        # Save image as png
                        # Reverse normalisation
                        image_tensor = data_sample['image_tensor']
                        image_tensor = image_tensor * torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
                        image_tensor = image_tensor + torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
                        image_tensor = image_tensor.permute(1, 2, 0).numpy()
                        image_tensor = (image_tensor * 255).astype(np.uint8)
                        image_tensor = Image.fromarray(image_tensor)
                        image_tensor.save(data_sample['image_path'].split('/')[-1].split('.')[0] + '.png')
                        if save_also_only_concept:
                            # Save image as png
                            # Reverse normalisation
                            image_tensor_only_concept = data_sample['image_tensor_only_concept']
                            image_tensor_only_concept = image_tensor_only_concept * torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
                            image_tensor_only_concept = image_tensor_only_concept + torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
                            image_tensor_only_concept = image_tensor_only_concept.permute(1, 2, 0).numpy()
                            image_tensor_only_concept = (image_tensor_only_concept * 255).astype(np.uint8)
                            image_tensor_only_concept = Image.fromarray(image_tensor_only_concept)
                            image_tensor_only_concept.save(data_sample['image_path'].split('/')[-1].split('.')[0] + '_only_concept.png')
                            if save_also_mask:
                                mask = data_sample['mask']
                                mask = mask.astype(np.uint8)
                                mask = Image.fromarray(mask)
                                mask.save(data_sample['image_path'].split('/')[-1].split('.')[0] + '_mask.png')
                        return
                    actual_id += 1
        else:
            for data_sample in self.List_data_image:
                if actual_id == idx:  
                    # Save image as png
                    # Reverse normalisation
                    image_tensor = data_sample['image_tensor']
                    image_tensor = image_tensor * torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
                    image_tensor = image_tensor + torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
                    image_tensor = image_tensor.permute(1, 2, 0).numpy()
                    image_tensor = (image_tensor * 255).astype(np.uint8)
                    image_tensor = Image.fromarray(image_tensor)
                    image_tensor.save(data_sample['image_path'].split('/')[-1])
                    return
                actual_id += 1
                
    def load_image_as_tensor(self, image_path,impait_part=None,return_mask=False):
        # Load the image and apply transformations
        image = Image.open(image_path).convert("RGB")
        if impait_part :
            if return_mask:
                image, mask = self.remove_sub_concept(image, impait_part, return_mask=return_mask)
                image_tensor = self.transform(image).to(self.device)
                # Resize mask 
                mask = cv2.resize(mask, (image_tensor.size(2), image_tensor.size(1)), interpolation=cv2.INTER_NEAREST)
                return image_tensor,mask
            else:
                image = self.remove_sub_concept(image, impait_part)
        '''image = image.resize((336, 336))'''
        
        image_tensor = self.transform(image)
        return image_tensor

    def remove_sub_concept(self,image, sub_concept, inflation_radius=1,return_mask=False):
        # Convert the image tensor back to a PIL image
        image_np = np.array(image)

        # Create an empty mask
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

        for coco_segmentation in sub_concept['segmentation']:
            # Convert segmentation points to a NumPy array
            polygon = np.array(coco_segmentation, dtype=np.int32).reshape(-1, 2)
            # Fill the mask with the polygon
            cv2.fillPoly(mask, [polygon], 255)

        # Inflate the mask by the specified radius
        kernel = np.ones((inflation_radius, inflation_radius), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        '''# only_concept the masked area
        modified_image = Image.fromarray(cv2.only_concept(image_np, dilated_mask, only_conceptRadius=5, flags=cv2.only_concept_TELEA))'''

        # Replace with white color
        modified_image = Image.fromarray(image_np)
        modified_image = Image.composite(modified_image, Image.new('RGB', modified_image.size, (255, 255, 255)), Image.fromarray(dilated_mask))

        if return_mask:
            # Resize mask 
            mask = cv2.resize(mask, (modified_image.size[1], modified_image.size[0]))
            return modified_image, dilated_mask

        return modified_image

    def compute_clip_scores(self, image_tensor, return_image_features=False,no_class=None):
        text_inputs = torch.cat([clip.tokenize(concept) for concept in self.clip_concepts]).to(self.device)

        if self.disantangled_version and self.weights is not None:
            similarity = torch.zeros(len(self.clip_concepts), device=self.device)
            for i,text_input in enumerate(text_inputs):
                with torch.no_grad():
                    self.prs.reinit()
                    image_features = self.model_clip.encode_image(image_tensor.unsqueeze(0).to(self.device), attn_method="head", normalize=False)
                    text_features = self.model_clip.encode_text(text_input.unsqueeze(0).to(self.device)).squeeze().detach().cpu().numpy()
                    attentions, _ = self.prs.finalize(image_features)
                    attentions = attentions.detach().cpu().numpy()
                    attn_map = attentions[0, :, 1:] @ text_features

                    # Old version, maybe rerun for ablation
                    '''if self.disantangled_version == 'object':

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
                        
                    elif self.disantangled_version == 'only_filter_register':
                        
                        L, N, H = attn_map.shape
                        filtered_attn = np.zeros_like(attn_map)
                        for l in range(L):
                            for h in range(H):
                                act = attn_map[l, :, h].reshape((self.image_size // self.patch_size, self.image_size // self.patch_size))
                                filtered = scipy.ndimage.median_filter(act, size=3)
                                filtered_attn[l, :, h] = filtered.reshape(-1)
                        similarity[i] = torch.tensor(filtered_attn).sum(axis=(0, 1, 2)).cpu()
                                
                    elif self.disantangled_version == 'only_weight':
                        
                        L, N, H = attn_map.shape
                        extended_weights = np.tile(self.weights[:, np.newaxis, :], (1, N, 1))
                        #extended_weights = np.tile(self.weights[no_class][:, np.newaxis, :], (1, N, 1))
                        weighted_attn = attn_map * extended_weights
                        similarity[i] = torch.tensor(weighted_attn).sum(axis=(0, 1, 2)).cpu()
         
                    elif self.disantangled_version == 'register':
                        
                        L, N, H = attn_map.shape
                        filtered_attn = np.zeros_like(attn_map)
                        for l in range(L):
                            for h in range(H):
                                act = attn_map[l, :, h].reshape((self.image_size // self.patch_size, self.image_size // self.patch_size))
                                filtered = scipy.ndimage.median_filter(act, size=3)
                                filtered_attn[l, :, h] = filtered.reshape(-1)

                        extended_weights = np.tile(self.weights[:, np.newaxis, :], (1, N, 1))
                        #extended_weights = np.tile(self.weights[no_class][:, np.newaxis, :], (1, N, 1))
                        weighted_attn = filtered_attn * (1-extended_weights)
                        similarity[i] = torch.tensor(weighted_attn).sum(axis=(0, 1, 2)).cpu()'''
                              
                    # New version
                    
                    if self.strategy == 'LTC' or self.strategy == 'loc':

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
                    
            clip_scores = {concept: similarity[i].item() for i, concept in enumerate(self.clip_concepts)}
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
                
            if return_image_features:
                return clip_scores, image_features

            return clip_scores

        else:
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

    def __len__(self):
        # Return the length of the dataset
        return len(self.List_data_image)

    def __getitem__(self, idx):
        # Return the data for a specific index
        
        if self.flag_import_in_getitem : 

            data = self.List_data_image[idx]
            # Load image as tensor

            image_tensor = self.load_image_as_tensor(data['image_tensor'])
            
            if self.train_cbm_mode:
                return {
                    'image_tensor': image_tensor,
                    'label_number': data['label_number']
                }
            
            return {
                'concepts': data['concepts'],
                'image_tensor': image_tensor,
                'mask': data['mask'],
                'concept_mask': data['concept_mask'],
                'label_number': data['label_number']
            }

        return self.List_data_image[idx]

# Example usage
# dataset = PIN_dataset(root_data='path/to/data', split='train', class_restrict=['kit_fox', 'English_setter'], clip_concepts=['dog', 'cat'])
