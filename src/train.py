from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO("yolo11n.yaml")  # or replace with "yolo11l.yaml" for a larger model

# Train the model
model.train(data="E://code/py/bsai/dataset.yaml", epochs=50)

results = model.val()
print(results)
