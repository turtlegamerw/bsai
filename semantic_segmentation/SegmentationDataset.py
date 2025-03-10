import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Define the SegmentationDataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, image_size=(224, 496)):
        self.image_dir = image_dir
        self.image_size = image_size
        
        # List all image and mask files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(image_dir) if f.startswith('mask_')])

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image and corresponding mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.image_dir, self.mask_files[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask as grayscale

        # Resize both image and mask
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)

        # Convert image to tensor
        image = transforms.ToTensor()(image)
        
        # Convert mask to tensor and make sure the mask values are integers
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# Create a function to visualize one image and its corresponding mask
def visualize_image_and_mask(dataset, idx=0):
    image, mask = dataset[idx]  # Get the image and mask for the given index
    
    # Convert the image tensor back to numpy for visualization
    image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

    # Plot the image and mask side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    # Display the mask
    ax[1].imshow(mask.numpy(), cmap='gray')  # Display the mask in grayscale
    ax[1].set_title("Mask")
    ax[1].axis("off")

    plt.show()


# Main part of the script
if __name__ == "__main__":
    # Set the path to your dataset
    image_dir = "D:/bsainewtraindataset"  # Path to your image and mask folder
    
    # Create the dataset object
    dataset = SegmentationDataset(image_dir)
    
    # Check if the dataset loaded correctly
    print(f"Total number of images and masks: {len(dataset)}")

    # Visualize the first image and its mask
    visualize_image_and_mask(dataset, idx=0)
