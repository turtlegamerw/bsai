import torch
from ultralytics import YOLO

# Load the checkpoint file
checkpoint = torch.load('src\\best.pt')

# Extract the model weights
model_weights = checkpoint['model']

# Initialize the model with the correct architecture
model = YOLO('yolov11n.pt')  # or the appropriate YOLO model

# Load the weights into the model
model.load_state_dict(model_weights)

# Now you can use the model for inference
print(model)
