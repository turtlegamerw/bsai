import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, image_size=(224, 496)):
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(image_dir) if f.startswith('mask_')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.image_dir, self.mask_files[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize images and masks to match target size
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)

        # Convert images and masks to tensors
        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

def train_model():
    # Hyperparameters and configurations
    image_dir = "D:/bsainewtraindataset"  # Path to your dataset folder
    num_epochs = 1  # Number of epochs
    batch_size = 8  # Batch size
    learning_rate = 0.001  # Learning rate
    save_interval = 1  # Save model every 'save_interval' epochs

    # Dataset and DataLoader
    dataset = SegmentationDataset(image_dir, image_size=(224, 496))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))  # Adjust output channels to match number of classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # For segmentation, typically cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            # Debugging: Check the unique mask values
            print("Unique mask values:", masks.unique())

            # Ensure the mask values are in the correct range
            masks = masks.clamp(min=0, max=1)

            # Ensure the mask is of type long
            masks = masks.long()

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)['out']  # Get model outputs
            print("Output shape:", outputs.shape)
            print("Mask shape:", masks.shape)

            loss = criterion(outputs, masks)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize the model

            running_loss += loss.item()

            # Print every 10th batch
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Print the average loss for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(dataloader):.4f}")

        # Save checkpoint every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

    # Save final model after all epochs
    torch.save(model.state_dict(), "final_model.pth")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train_model()
