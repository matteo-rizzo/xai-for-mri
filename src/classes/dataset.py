from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

from src.config import PATH_TO_MASKS


class MRIDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        """
        Custom Dataset for MRI images and corresponding masks.
        :param data: List of tuples containing image paths and labels.
        :param transform: Transformation to apply to the images.
        :param target_transform: Transformation to apply to the labels.
        """
        self.img_labels = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]

        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Construct mask path
        mask_path = Path(PATH_TO_MASKS) / img_path.split('\\')[-2] / img_path.split('\\')[-1]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)  # Convert mask to binary (0 or 1)

        # Pad image and mask to square shape
        square_img = self.pad_to_square(img)
        square_mask = self.pad_to_square(mask)

        if self.transform:
            square_img = self.transform(square_img)
        if self.target_transform:
            label = self.target_transform(label)

        return square_img, label, square_mask

    @staticmethod
    def pad_to_square(img, target_size=224):
        """
        Pad image to a square of the larger side, ensuring a minimum size.
        :param img: Input image
        :param target_size: Minimum size for the image
        :return: Padded square image
        """
        s = max(img.shape[:2])
        s = max(s, target_size)  # Ensure minimum size of target_size

        # Create a black square image
        square_img = np.zeros((s, s), np.uint8)
        ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2  # Centering position
        square_img[ay:ay + img.shape[0], ax:ax + img.shape[1]] = img

        return square_img


class MRISubset(Dataset):
    def __init__(self, subset, train_bool=False, transform=None):
        """
        Subset of the MRI dataset with optional transformations and training-specific augmentations.
        :param subset: Subset of the dataset.
        :param train_bool: Flag indicating whether the subset is for training (enables augmentations).
        :param transform: Transformation to apply to the images.
        """
        self.subset = subset
        self.transform = transform
        self.train_bool = train_bool

    def __getitem__(self, index):
        x, y, mask = self.subset[index]
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Convert mask to tensor and add dimension

        if self.transform:
            x = self.transform(x)

        if self.train_bool:
            x = self.apply_data_augmentation(x, mask_tensor)
        else:
            x = x * mask_tensor

        return x, y  # Keep the black background

    def __len__(self):
        return len(self.subset)

    @staticmethod
    def apply_data_augmentation(x, mask_tensor):
        """
        Apply data augmentation techniques for training.
        :param x: Image tensor
        :param mask_tensor: Mask tensor
        :return: Augmented image tensor
        """
        if torch.rand(1) < 0.5:
            hflip = transforms.RandomHorizontalFlip(p=1.0)  # Random horizontal flip (also applied to mask)
            x = hflip(x * mask_tensor)
        else:
            x = x * mask_tensor

        return x
