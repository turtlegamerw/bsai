import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from unet_model import get_model

# Inference function
def predict_image(image_path, model):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (256, 256)) / 255.0  # Normalize
    image_input = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    image_input = torch.tensor(image_input).permute(0, 3, 1, 2).float().cuda()

    model.eval()
    with torch.no_grad():
        predicted_mask = model(image_input)
    
    predicted_mask = torch.sigmoid(predicted_mask)
    predicted_mask = (predicted_mask > 0.5).cpu().numpy().astype(np.uint8)  # Convert to binary

    return predicted_mask[0][0]  # Return the mask

# Example usage
if __name__ == "__main__":
    model = get_model().cuda()
    model.load_state_dict(torch.load('unet_model.pth'))

    image_path = 'path/to/new_image.png'
    predicted_mask = predict_image(image_path, model)

    plt.imshow(predicted_mask, cmap='gray')
    plt.show()
