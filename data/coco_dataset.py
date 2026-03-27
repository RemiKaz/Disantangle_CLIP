import json
import os
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable as V
from torch.nn import functional as F
import cv2
from clip_text_span.utils.factory import get_tokenizer
import clip
import scipy
from clip_text_span.prs_hook import hook_prs_logger

class COCODataset(Dataset):
    """A custom dataset class to load and process images and annotations from the COCO dataset."""
    def __init__(self, root_data='/lustre/fsmisc/dataset/COCO', root_parts='/lustre/fsmisc/dataset/COCO', split='train', pth_metadata=None,
                 pth_metadata_parts=None, select_segmentation='full', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                 reduce_samples=False, train_cbm_mode=False, clip_concepts=None, class_restrict=None,disantangled_version=False):
        """
        Initialize the COCO dataset. Load metadata and COCO annotations to prepare image and segmentation data.
        """
        super(COCODataset, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.root_data = root_data
        self.root_parts = root_parts
        self.split = split
        self.model_clip_name = model_name + pretrained
        self.metadata = self.load_metadata(pth_metadata) if pth_metadata else {}
        self.metadata_parts = json.load(open(pth_metadata_parts)) if pth_metadata_parts else {}
        self.select_segmentation = select_segmentation
        self.reduce_samples = reduce_samples
        self.train_cbm_mode = train_cbm_mode
        self.clip_concepts = clip_concepts
        self.class_restrict = class_restrict
        self.disantangled_version = disantangled_version

        # Initialize the CLIP model and transforms if needed
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust based on model requirements
            transforms.ToTensor(),
        ])

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

        ## To assign a class to each image TODO make metadata_file
        shopping_and_dining = ['bakery/shop', 'ballroom', 'bar', 'bazaar/indoor', 'beauty_salon', 'beer_hall', 'bookstore', 'bow_window/indoor', 'bowling_alley', 'candy_store', 'cafeteria', 'butchers_shop', 'clean_room', 'closet', 'clothing_store', 'coffee_shop', 'conference_center', 'conference_room', 'delicatessen', 'department_store', 'dining_hall', 'dining_room', 'discotheque', 'drugstore',  'florist_shop/indoor', 'food_court', 'galley', 'general_store/indoor', 'gift_shop', 'greenhouse/indoor', 'ice_cream_parlor', 'jewelry_shop', 'laundromat', 'market/indoor', 'movie_theater/indoor', 'pizzeria', 'pub/indoor', 'reception', 'repair_shop', 'restaurant', 'restaurant_kitchen', 'sauna', 'shoe_shop', 'shopping_mall/indoor', 'sushi_bar', 'ticket_booth', 'toyshop', 'wet_bar','bank_vault', 'banquet_hall', 'basement', 'bathroom', 'berth', 'burial_chamber', 'catacomb', 'cockpit', 'dorm_room', 'dressing_room', 'entrance_hall', 'fabric_store', 'flea_market/indoor', 'lobby', 'locker_room', 'parking_garage/indoor', 'pet_shop', 'pharmacy', 'stable', 'supermarket','fastfood_restaurant']
        workplace = ['assembly_line', 'auto_factory', 'auto_showroom', 'biology_laboratory', 'chemistry_lab', 'computer_room', 'engine_room', 'garage/indoor', 'hangar/indoor', 'hospital_room', 'office', 'office_cubicles', 'operating_room', 'server_room', 'utility_room', 'veterinarians_office','atrium/public', 'auditorium', 'corridor', 'hardware_store', 'jail_cell', 'legislative_chamber']
        home_or_hotel = ['alcove', 'artists_loft', 'attic', 'bedchamber', 'bedroom', 'booth/indoor', 'childs_room', 'home_office', 'home_theater', 'hotel_room', 'jacuzzi/indoor', 'kitchen', 'living_room', 'mezzanine', 'nursery', 'nursing_home', 'pantry', 'recreation_room', 'shower', 'stage/indoor', 'staircase', 'storage_room', 'television_room', 'throne_room', 'waiting_room', 'youth_hostel']
        transportation = ['airplane_cabin', 'airport_terminal', 'bus_interior', 'car_interior', 'elevator/door', 'elevator_lobby', 'elevator_shaft', 'escalator/indoor', 'subway_station/platform', 'train_interior', 'train_station/platform','amusement_arcade']
        sports_and_leisure = ['arena/hockey', 'arena/performance', 'arena/rodeo', 'ball_pit', 'basketball_court/indoor', 'boxing_ring', 'ice_skating_rink/indoor', 'martial_arts_gym', 'playroom', 'swimming_hole', 'swimming_pool/indoor','gymnasium/indoor']
        cultural = ['aquarium', 'arcade', 'archaelogical_excavation', 'archive', 'art_gallery', 'art_school', 'art_studio', 'church/indoor', 'classroom', 'kindergarden_classroom', 'lecture_room', 'library/indoor', 'museum/indoor', 'music_studio', 'natural_history_museum', 'orchestra_pit', 'physics_laboratory', 'science_museum', 'television_studio']
        self.All = ['airplane_cabin','airport_terminal','alcove','amusement_arcade','aquarium','arcade','archaelogical_excavation','archive','arena/hockey','arena/performance','arena/rodeo','art_gallery','art_school','art_studio','artists_loft','assembly_line','atrium/public','attic','auditorium','auto_factory','auto_showroom','bakery/shop','ball_pit','ballroom','bank_vault','banquet_hall','bar','basement','basketball_court/indoor','bathroom','bazaar/indoor','beauty_salon','bedchamber','bedroom','beer_hall','berth','biology_laboratory','bookstore','booth/indoor','bow_window/indoor','bowling_alley','boxing_ring','burial_chamber','bus_interior','butchers_shop','cafeteria','candy_store','car_interior','catacomb','chemistry_lab','childs_room','church/indoor','classroom','clean_room','closet','clothing_store','cockpit','coffee_shop','computer_room','conference_center','conference_room','corridor','delicatessen','department_store','dining_hall','dining_room','discotheque','dorm_room','dressing_room','drugstore','elevator/door','elevator_lobby','elevator_shaft','engine_room','entrance_hall','escalator/indoor','fabric_store','fastfood_restaurant','flea_market/indoor','florist_shop/indoor','food_court','galley','garage/indoor','general_store/indoor','gift_shop','greenhouse/indoor','gymnasium/indoor','hangar/indoor','hardware_store','home_office','home_theater','hospital_room','hotel_room','ice_cream_parlor','ice_skating_rink/indoor','jacuzzi/indoor','jail_cell','jewelry_shop','kindergarden_classroom','kitchen','laundromat','lecture_room','legislative_chamber','library/indoor','living_room','lobby','locker_room','market/indoor','martial_arts_gym','mezzanine','movie_theater/indoor','museum/indoor','music_studio','natural_history_museum','nursery','nursing_home','office','office_cubicles','operating_room','orchestra_pit','pantry','parking_garage/indoor','pet_shop','pharmacy','physics_laboratory','pizzeria','playroom','pub/indoor','reception','recreation_room','repair_shop','restaurant','restaurant_kitchen','sauna','science_museum','server_room','shoe_shop','shopping_mall/indoor','shower','stable','stage/indoor','staircase','storage_room','subway_station/platform','supermarket','sushi_bar','swimming_hole','swimming_pool/indoor','television_room','television_studio','throne_room','ticket_booth','toyshop','train_interior','train_station/platform','utility_room','veterinarians_office','waiting_room','wet_bar','youth_hostel']

        self.D_classes = {}

        for classes in shopping_and_dining:
            self.D_classes[classes] = 0
            
        for classes in workplace:
            self.D_classes[classes] = 1
            
        for classes in home_or_hotel:
            self.D_classes[classes] = 2
            
        for classes in transportation:
            self.D_classes[classes] = 3
            
        for classes in sports_and_leisure:
            self.D_classes[classes] = 4
            
        for classes in cultural:
            self.D_classes[classes] = 5

        self.metadata_parts = {
            "shopping_and_dining": [
                "backpack", "handbag", "suitcase", "wine glass", "cup", "sandwich", "pizza", "donut", "cake",
                "microwave", "oven", "refrigerator", "book", "clock", "vase"
            ],
            "workplace": [
                "laptop", "mouse", "keyboard", "cell phone", "book", "clock", "chair", "dining table", "remote"
            ],
            "home_or_hotel": [
                "bed", "couch", "potted plant", "toilet", "tv", "laptop", "mouse", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ],
            "transportation": [
                "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter"
            ],
            "sports_and_leisure": [
                "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket"
            ],
            "cultural": [
                "book", "clock", "vase"
            ]
        }
    
        self.id_to_category_name = {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            5: 'airplane',
            6: 'bus',
            7: 'train',
            8: 'truck',
            9: 'boat',
            10: 'traffic light',
            11: 'fire hydrant',
            13: 'stop sign',
            14: 'parking meter',
            15: 'bench',
            16: 'bird',
            17: 'cat',
            18: 'dog',
            19: 'horse',
            20: 'sheep',
            21: 'cow',
            22: 'elephant',
            23: 'bear',
            24: 'zebra',
            25: 'giraffe',
            27: 'backpack',
            28: 'umbrella',
            31: 'handbag',
            32: 'tie',
            33: 'suitcase',
            34: 'frisbee',
            35: 'skis',
            36: 'snowboard',
            37: 'sports ball',
            38: 'kite',
            39: 'baseball bat',
            40: 'baseball glove',
            41: 'skateboard',
            42: 'surfboard',
            43: 'tennis racket',
            44: 'bottle',
            46: 'wine glass',
            47: 'cup',
            48: 'fork',
            49: 'knife',
            50: 'spoon',
            51: 'bowl',
            52: 'banana',
            53: 'apple',
            54: 'sandwich',
            55: 'orange',
            56: 'broccoli',
            57: 'carrot',
            58: 'hot dog',
            59: 'pizza',
            60: 'donut',
            61: 'cake',
            62: 'chair',
            63: 'couch',
            64: 'potted plant',
            65: 'bed',
            67: 'dining table',
            70: 'toilet',
            72: 'tv',
            73: 'laptop',
            74: 'mouse',
            75: 'remote',
            76: 'keyboard',
            77: 'cell phone',
            78: 'microwave',
            79: 'oven',
            80: 'toaster',
            81: 'sink',
            82: 'refrigerator',
            84: 'book',
            85: 'clock',
            86: 'vase',
            87: 'scissors',
            88: 'teddy bear',
            89: 'hair drier',
            90: 'toothbrush',
        }

        self.labels = [
            "shopping_and_dining",
            "workplace",
            "home_or_hotel",
            "transportation",
            "sports_and_leisure",
            "cultural",
        ]

        # Map classes to IDs
        for i, classes in enumerate([shopping_and_dining, workplace, home_or_hotel,
                                    transportation, sports_and_leisure, cultural]):
            for cls in classes:
                self.D_classes[cls] = i

        # load the class label for Places365
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        self.classes_places = list()
        with open(file_name) as class_file:
            for line in class_file:
                self.classes_places.append(line.strip().split(' ')[0][3:])
        self.classes_places = tuple(self.classes_places)

        # Load the model to create pseudo labels
        # Init the places CNN for pseudo labeling
        # the architecture to use
        arch = 'resnet18'
        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        self.model_places = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model_places.load_state_dict(state_dict)
        self.model_places.eval()

        # load the image transformer
        self.transform_places = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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
                self.weights = np.mean(np.load(f'weights/weights_coco_{self.method_iou}_{self.type_concept}_{self.strategy}_all_classes.npy'),axis=0)  # Load weights for disentangled version
                
        # Load dataset
        self.load_data()

    def load_metadata(self, pth_metadata):
        """Load metadata from a JSON file."""
        if pth_metadata:
            with open(pth_metadata) as f:
                return json.load(f)
        return {}

    def load_data(self):
        """Load and process COCO dataset images and annotations."""
        self.List_data_image = []
        coco = COCO(os.path.join(self.root_parts, f'annotations/instances_{self.split}2017.json'))
        img_ids = coco.getImgIds()
        
        if self.reduce_samples:
            slice_val = 500
        else:
            slice_val = None 

        for img_id in tqdm(img_ids[:slice_val], desc="Loading COCO data"):
            
            if self.class_restrict:
                for class_name in self.class_restrict:
                    img_info = coco.loadImgs(img_id)[0]

                    image_path = os.path.join(self.root_data, self.split + '2017', img_info['file_name'])
                    
                    image = Image.open(image_path).convert("RGB")

                    # Get class prediction from Places365 model
                    input_img = V(self.transform_places(image).unsqueeze(0))
                    logit = self.model_places.forward(input_img)
                    h_x = F.softmax(logit, 1).data.squeeze()
                    _, idx = h_x.sort(0, True)
                    class_pred = self.classes_places[idx[0]]
                    
                    if class_pred not in self.All : # Check if img classified as indoor
                        continue
                    
                    else: 
                        class_id = int(self.D_classes[class_pred])
    
                    label_name = self.labels[class_id]
                  
                    if label_name not in class_name:
                        continue          
          
                    image_tensor = self.transform(image).to(self.device)
                    
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    anns = coco.loadAnns(ann_ids)

                    if '-' in class_name:
                        parts = class_name.split('-')
                        
                        if parts[1] == 'random': # Select a random part in the class of the image and save it if present
                            part_to_use = np.random.choice(self.metadata_parts[class_id])
                            sub_concept = {'name_removed':part_to_use,'segmentation':[],'coco':coco}
                            
                        else:
                            sub_concept = {'name_removed':parts[1],'segmentation':[],'coco':coco}

                        segmentation_data = []
                        category_maps = {}
                        for ann in anns:
                            category_name = coco.loadCats(int(ann['category_id']))[0]["name"].replace(' ', '_')
                            if category_name == sub_concept['name_removed']:
                                sub_concept['segmentation'].append(ann['segmentation'])
                    else:
                        # Handle full segmentation
                        sub_concept = {'name_removed': 'all', 'segmentation': [ann['segmentation'] for ann in anns], 'coco': coco}
                    
                    # Match concepts
                    concepts = [coco.loadCats(int(ann['category_id']))[0]["name"].replace(' ', '_') for ann in anns]
                    segmentation_data = [[ann['category_id'], ann['segmentation']] for ann in anns]

                    category_maps = {ann['category_id']: coco.loadCats(int(ann['category_id']))[0]["name"] for ann in anns}

                    if self.select_segmentation != 'full':
                        image_tensor_only_concept, mask = self.load_image_as_tensor(image_path, impait_part=sub_concept, return_mask=True)

                        if self.clip_concepts:
                            clip_scores, image_features = self.compute_clip_scores(image_tensor, return_image_features=True)
                            clip_scores_only_concept, image_features_only_concept = self.compute_clip_scores(
                                image_tensor_only_concept, return_image_features=True)
                        else:
                            clip_scores = None
                            clip_scores_only_concept = None
                            image_features = None
                            image_features_only_concept = None

                        self.List_data_image.append({
                            'concepts': concepts,
                            'label_number': class_id,
                            'class_name': label_name,
                            'image_path': image_path,
                            'image_tensor': image_tensor,
                            'image_tensor_only_concept': image_tensor_only_concept,
                            'image_features': image_features,
                            'image_features_only_concept': image_features_only_concept,
                            'segmentation_data': segmentation_data,
                            'concept_mask': sub_concept['name_removed'],
                            'mask': mask,
                            'clip_scores': clip_scores,
                            'clip_scores_only_concept': clip_scores_only_concept,
                            'catIds': category_maps
                        })
                    else:

                        if self.clip_concepts:
                            clip_scores, image_features = self.compute_clip_scores(image_tensor, return_image_features=True)
                        else:
                            clip_scores = None
                            image_features = None

                        _, mask = self.load_image_as_tensor(image_path, impait_part=sub_concept, return_mask=True)

                        self.List_data_image.append({
                            'concepts': concepts,
                            'label_number': class_id,
                            'class_name': label_name,
                            'image_path': image_path,
                            'image_tensor': image_tensor,
                            'segmentation_data': segmentation_data,
                            'concept_mask': sub_concept['name_removed'],
                            'mask': mask,
                            'clip_scores': clip_scores,
                            'catIds': category_maps
                        })
                
            else:
                img_info = coco.loadImgs(img_id)[0]
                image_path = os.path.join(self.root_data, self.split + '2017', img_info['file_name'])

                # Load and preprocess image
                image = Image.open(image_path).convert("RGB")

                # Get class prediction from Places365 model
                input_img = V(self.transform_places(image).unsqueeze(0))
                logit = self.model_places.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                _, idx = h_x.sort(0, True)
                class_pred = self.classes_places[idx[0]]

                # Skip if not in our predefined categories
                if class_pred not in self.D_classes:
                    continue

                # Get the class ID from Places365 prediction
                class_id = self.D_classes.get(class_pred, -1)
                if class_id == -1:
                    continue  # Skip if class not found in our mapping

                # Get the category name based on class_id
                if class_id < len(self.labels):
                    category_name = self.labels[class_id]
                else:
                    category_name = "unknown"

                # Get image tensor using CLIP transform
                image_tensor = self.transform(image).to(self.device)

                # Get annotations from COCO
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)

                # Handle full segmentation
                sub_concept = {'name_removed': 'all', 'segmentation': [ann['segmentation'] for ann in anns], 'coco': coco}

                # Get concepts and segmentation data
                concepts = [coco.loadCats(int(ann['category_id']))[0]["name"].replace(' ', '_') for ann in anns]
                segmentation_data = [[ann['category_id'], ann['segmentation']] for ann in anns]
                category_maps = {ann['category_id']: coco.loadCats(int(ann['category_id']))[0]["name"] for ann in anns}

                # Load image with potential masking
                if self.select_segmentation != 'full':
                    image_tensor_only_concept, mask = self.load_image_as_tensor(image_path, impait_part=sub_concept, return_mask=True)

                    # Compute CLIP scores if needed
                    if self.clip_concepts:
                        clip_scores, image_features = self.compute_clip_scores(image_tensor, return_image_features=True)
                        clip_scores_only_concept, image_features_only_concept = self.compute_clip_scores(
                            image_tensor_only_concept, return_image_features=True)
                    else:
                        clip_scores = None
                        clip_scores_only_concept = None
                        image_features = None
                        image_features_only_concept = None

                    # Append data with concept-specific information
                    self.List_data_image.append({
                        'concepts': anns,
                        'label_number': class_id,
                        'class_name': category_name,
                        'image_path': image_path,
                        'image_tensor': image_tensor,
                        'image_tensor_only_concept': image_tensor_only_concept,
                        'image_features': image_features,
                        'image_features_only_concept': image_features_only_concept,
                        'segmentation_data': segmentation_data,
                        'concept_mask': sub_concept['name_removed'],
                        'mask': mask,
                        'clip_scores': clip_scores,
                        'clip_scores_only_concept': clip_scores_only_concept,
                        'catIds': category_maps
                    })
                else:
                    
                    img_info = coco.loadImgs(img_id)[0]
                    
                    # Get annotations from COCO
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    anns = coco.loadAnns(ann_ids)
                    
                    # Append data without concept-specific information
                    if self.clip_concepts:
                        clip_scores, image_features = self.compute_clip_scores(image_tensor, return_image_features=True)
                    else:
                        clip_scores = None
                        image_features = None

                    # Create a mask for the full image
                    _, mask = self.load_image_as_tensor(image_path, impait_part=sub_concept, return_mask=True)

                    self.List_data_image.append({
                        'concepts': anns,
                        'label_number': class_id,
                        'class_name': category_name,
                        'image_path': image_path,
                        'image_tensor': image_tensor,
                        'segmentation_data': segmentation_data,
                        'concept_mask': sub_concept['name_removed'],
                        'mask': mask,
                        'clip_scores': clip_scores,
                        'catIds': category_maps
                    })

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

    def load_image_as_tensor(self, image_path, impait_part=None, return_mask=False):
        # Load the image and apply transformations
        image = Image.open(image_path).convert("RGB")
        if impait_part:
            if return_mask:
                image, mask = self.remove_sub_concept(image, impait_part, return_mask=return_mask)
                image_tensor = self.transform(image).to(self.device)
                # Resize mask
                mask = cv2.resize(mask, (image_tensor.size(2), image_tensor.size(1)), interpolation=cv2.INTER_NEAREST)
                return image_tensor, mask
            else:
                image = self.remove_sub_concept(image, impait_part)
        '''
        image = image.resize((336, 336))
        '''
        image_tensor = self.transform(image)
        return image_tensor.to(self.device)

    def rle_decode(self, counts, height, width):
        """
        Decode the RLE format into a binary mask, flipping the position trajectory from horizontal to vertical.
        Args:
            counts: A list of counts representing start and length of runs.
            height: Height of the image/mask.
            width: Width of the image/mask.
        Returns:
            Binary mask as a NumPy array.
        """
        binary_mask = np.zeros(height * width, dtype=np.uint8)
        flag_mask = False
        position = 0
        for i in range(len(counts)):
            if flag_mask:
                binary_mask[position:position + counts[i]] = 1
            position += counts[i]
            flag_mask = not flag_mask
        # Reshape to (width, height) and then transpose to (height, width)
        binary_mask = binary_mask.reshape((width, height)).T
        return binary_mask

    def remove_sub_concept(self, image, sub_concept, inflation_radius=1, return_mask=False):
        # Convert the image tensor back to a PIL image
        image_np = np.array(image)
        # Create an empty mask
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

        for coco_segmentation in sub_concept['segmentation']:
            if isinstance(coco_segmentation, list):  # Polygon format
                for sub_seg in coco_segmentation:
                    polygon = np.array(sub_seg, dtype=np.int32).reshape(-1, 2)
                    # Fill the mask with the polygon
                    cv2.fillPoly(mask, [polygon], 255)
            else:  # RLE format
                if 'counts' in coco_segmentation:
                    rle_data = coco_segmentation
                    counts = rle_data['counts']
                    height, width = rle_data['size']
                    # Decode and apply the RLE
                    binary_mask = self.rle_decode(counts, height, width)
                    # Resize if needed to match image dimensions
                    if binary_mask.shape != image_np.shape[:2]:
                        binary_mask = cv2.resize(binary_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Use logical OR to add the RLE mask to the current mask
                    mask = np.logical_or(mask, binary_mask).astype(np.uint8) * 255

        # Inflate the mask by the specified radius
        kernel = np.ones((inflation_radius, inflation_radius), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Replace masked area with white color
        modified_image = Image.fromarray(image_np)
        modified_image = Image.composite(
            modified_image,
            Image.new('RGB', modified_image.size, (255, 255, 255)),
            Image.fromarray(dilated_mask)
        )

        if return_mask:
            # Resize mask
            mask = cv2.resize(mask, (modified_image.size[0], modified_image.size[1]))
            return modified_image, dilated_mask
        return modified_image

    def process_annotations(self, coco, img_id):
        """Process COCO annotations to extract concepts and segmentation data."""
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        concepts = []
        segmentation_data = []
        for ann in anns:
            category = coco.loadCats(ann['category_id'])[0]
            concepts.append(category['name'])
            segmentation_data.append(ann['segmentation'])
        return concepts, segmentation_data

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
            'image_tensor_only_concept': data.get('image_tensor_only_concept'),
            'image_features': data.get('image_features'),
            'image_features_only_concept': data.get('image_features_only_concept'),
            'segmentation_data': data['segmentation_data'],
            'concepts': data['concepts'],
            'label_number': data['label_number'],
            'catIds': data['catIds'],
            'clip_scores': data.get('clip_scores'),
            'clip_scores_only_concept': data.get('clip_scores_only_concept'),
            'concept_mask': data['concept_mask'],
            'mask': data.get('mask')
        }
