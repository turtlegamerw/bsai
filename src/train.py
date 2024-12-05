from ultralytics import YOLO

if __name__ == "__main__":
    # Load a YOLOv11 model
    model = YOLO("yolo11n.yaml")
    
    # Train the model
    model.train(data="E://code/py/bsai/dataset.yaml", epochs=50, device=0)
    
    # Validate the model
    results = model.val()
    print(results)
