import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_file, save_dir, image_size=(224, 496)):
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.image_size = image_size
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load annotations from JSON
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Collect image file names
        self.image_files = [img['file_name'] for img in self.annotations['images']]
        
        # Create a mapping of image_id to annotations
        self.annotations_dict = {img['id']: [] for img in self.annotations['images']}
        for ann in self.annotations['annotations']:
            self.annotations_dict[ann['image_id']].append(ann)
        
        # Store category information
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = f"{self.image_dir}/{img_file}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Resize the image while maintaining aspect ratio
        scale_x = self.image_size[1] / float(orig_width)
        scale_y = self.image_size[0] / float(orig_height)
        scale = min(scale_x, scale_y)

        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image_resized = cv2.resize(image, (new_width, new_height))

        # Create padding to match the target size
        top_padding = (self.image_size[0] - new_height) // 2
        bottom_padding = self.image_size[0] - new_height - top_padding
        left_padding = (self.image_size[1] - new_width) // 2
        right_padding = self.image_size[1] - new_width - left_padding

        # Add padding to the image
        image_padded = cv2.copyMakeBorder(image_resized, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Get annotations for this image
        img_id = idx  # Assuming image IDs are sequential
        annotations = self.annotations_dict.get(img_id, [])
        
        # Create a mask (same size as image) filled with zeros
        mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)

        # Loop through annotations and draw bounding boxes on the mask
        for ann in annotations:
            category_id = ann['category_id']
            # Resize the bounding box to match the resized image size
            x, y, w, h = ann['bbox']
            
            # Ensure the bounding box stays within the image bounds
            x, y = max(0, x), max(0, y)  # Ensure x, y are not negative
            w, h = min(self.image_size[1] - x, w), min(self.image_size[0] - y, h)  # Ensure w, h don't overflow

            # Resize the bounding box to match the resized image size
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            
            # Draw the bounding box on the mask for the current category
            mask[int(y):int(y+h), int(x):int(x+w)] = category_id

        # Save the image and mask
        img_save_path = os.path.join(self.save_dir, f"image_{idx}.jpg")
        mask_save_path = os.path.join(self.save_dir, f"mask_{idx}.png")
        
        cv2.imwrite(img_save_path, cv2.cvtColor(image_padded, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_save_path, mask)

        # Convert mask to tensor format
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension (1 for grayscale)
        return image_padded, mask

# Example usage
image_dir = "D:/bsaitraindata/train"
annotation_file = "D:/bsaitraindata/train/_annotations.coco.json"
save_dir = "D:/bsainewtraindataset"

dataset = SegmentationDataset(image_dir, annotation_file, save_dir)

# Iterate over the dataset and process images
for i in tqdm(range(len(dataset))):
    image, mask = dataset[i]
    # You can do further processing here if needed
