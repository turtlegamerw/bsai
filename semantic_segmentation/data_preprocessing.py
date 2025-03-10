import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Function to load images and masks from a directory
def load_data(image_dir, mask_dir, image_size=(256, 256)):
    images = []
    masks = []
    
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for image_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(image_dir, image_file))
        img = cv2.resize(img, image_size)
        images.append(img)

        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        masks.append(mask)
    
    images = np.array(images) / 255.0  # Normalize to [0, 1]
    masks = np.array(masks) / 255.0    # Normalize to [0, 1]
    
    return images, masks

# Example usage
if __name__ == "__main__":
    image_dir = 'path/to/images'
    mask_dir = 'path/to/masks'
    images, masks = load_data(image_dir, mask_dir)
    
    # Split into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
