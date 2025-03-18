import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import cv2
import mss
import pygetwindow as gw
import numpy as np

# Load the trained model
model = models.segmentation.deeplabv3_resnet101(pretrained=False)  # Initialize the model
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))  # Adjust output channels for 2 classes

# Load the model weights, ignoring unexpected keys
model.load_state_dict(torch.load('final_model.pth'), strict=False)  # Use strict=False to ignore missing or unexpected keys
model.eval()  # Set the model to evaluation mode

# Define the transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per pretrained model
])

# Find the scrcpy window (for cropping)
window = gw.getWindowsWithTitle('SM-A356B')[0]

# Initialize screen capture using mss
with mss.mss() as sct:
    while True:
        # Define screen region based on scrcpy window
        bbox = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height
        }
        screenshot = sct.grab(bbox)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Preprocess the frame for the model
        input_image = cv2.resize(frame, (224, 496))  # Resize to match model input size
        input_tensor = preprocess(input_image)  # Preprocess the image
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        # Move the input to the same device as the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_batch = input_batch.to(device)
        model.to(device)

        # Perform inference
        with torch.no_grad():  # No need to track gradients for inference
            output = model(input_batch)['out'][0]  # Get output from model
            output_predictions = output.argmax(0)  # Get the class with the highest score for each pixel

        # Post-process the output
        output_predictions = output_predictions.cpu().numpy()  # Move to CPU for visualization

        # Resize the output mask to the original frame size
        output_predictions_resized = cv2.resize(output_predictions, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Visualize the result (for example, by coloring detected regions)
        output_mask = np.zeros_like(frame)
        output_mask[output_predictions_resized == 1] = [0, 255, 0]  # Green for class 1 (adjust if necessary)

        # Check if the mask has non-zero values
        if np.any(output_mask != 0):
            print("Mask has non-zero values")
        else:
            print("Mask is empty")

        # Show the mask image
        cv2.imshow("Mask", output_mask)

        # Combine the original frame with the mask (overlay)
        frame_with_overlay = cv2.addWeighted(frame, 1, output_mask, 1, 0)

        # Check if the mask is applied correctly
        cv2.imshow("Real-time Segmentation", frame_with_overlay)

        # Wait for a key event to keep the window open (add a small delay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up after finishing
cv2.destroyAllWindows()
