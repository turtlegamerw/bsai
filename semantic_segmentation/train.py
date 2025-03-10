import torch
import torch.optim as optim
import torch.nn as nn
from unet_model import get_model
from data_preprocessing import load_data
from tqdm import tqdm

# Training function
def train_model(model, train_loader, num_epochs=10, lr=1e-4):
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), 'unet_model.pth')

# Example usage
if __name__ == "__main__":
    image_dir = 'path/to/images'
    mask_dir = 'path/to/masks'

    train_loader = load_data(image_dir, mask_dir, batch_size=4)

    model = get_model().cuda()  # Move model to GPU if available
    train_model(model, train_loader)
