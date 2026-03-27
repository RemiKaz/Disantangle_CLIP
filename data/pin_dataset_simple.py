import torch
import os
from torchvision import transforms
from PIL import Image

class PartImageNetSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """Load the processed PartImageNet dataset

        Args:
            root (str): Path to root directory
            transform (optional): Transformations to apply to the images.
                Defaults to None.
        """
        self.root = root
        self.img_path = os.path.join(self.root, "img")
        self.seg_path = os.path.join(self.root, "seg")

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to a fixed size
            transforms.ToTensor(),
        ])

        self.classes = self._list_classes(self.img_path)
        self.num_classes = len(self.classes)

        # Assuming CLASSES is defined somewhere in your environment
        if self.num_classes == 1000:
            if "partimagenetpp" in self.root:
                from .classinfo_1k_pp import CLASSES

        self.classes_list = list(CLASSES.keys())
        self.num_seg_labels = sum([CLASSES[c] for c in self.classes])

        self.images, self.masks = self._get_data()

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _mask = Image.open(self.masks[index])

        if self.transform is not None:
            _img = self.transform(_img)
            _mask = self.transform(_mask)

        # Convert mask to binary mask (concatenation of all parts)
        _mask = (_mask > 0).float()

        # Return only the required entries
        return {
            'concepts': self.classes_list[self._get_class_index(self.images[index])],
            'image_tensor': _img,
            'mask': _mask
        }

    def _get_data(self):
        images, masks = [], []
        for label in self.classes:
            c_img_path = os.path.join(self.img_path, label)
            c_part_path = os.path.join(self.seg_path, label)

            imglist = os.listdir(c_img_path)
            images.extend([os.path.join(c_img_path, name) for name in imglist])
            masks.extend([os.path.join(c_part_path, name.split(".")[0] + ".png") for name in imglist])

        return images, masks

    def _list_classes(self, directory=None):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        return classes

    def _get_class_index(self, image_path):
        return self.classes.index(image_path.split('/')[-2])

    def __len__(self):
        return len(self.images)