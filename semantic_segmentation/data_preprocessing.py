import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize(image_size),  # Resize images
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.image_files[idx]))
        img = self.transform(img)

        mask = cv2.imread(os.path.join(self.mask_dir, self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        mask = self.transform(mask)
        mask = mask.unsqueeze(0)  # Add channel dimension to the mask

        return img, mask

# Example usage
def load_data(image_dir, mask_dir, batch_size=4):
    dataset = SegmentationDataset(image_dir, mask_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader
