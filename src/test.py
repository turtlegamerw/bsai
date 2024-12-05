import torch
import torchvision
print(torch.__version__)  # PyTorch version
print(torchvision.__version__)  # TorchVision version
print(torch.cuda.is_available())  # Check if CUDA is detected
print(torch.version.cuda)  # Installed CUDA version
