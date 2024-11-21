import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
from ultralytics import YOLO
import torch

# Find the Scrcpy window
scrcpy_windows = [win for win in gw.getWindowsWithTitle('RMX1911') if win.visible]

if not scrcpy_windows:
    print("Scrcpy window not found. Make sure Scrcpy is running.")
    exit()

# Get Scrcpy window coordinates
scrcpy_window = scrcpy_windows[0]
bbox = {
    "top": scrcpy_window.top,
    "left": scrcpy_window.left,
    "width": scrcpy_window.width,
    "height": scrcpy_window.height
}

# Load your own YOLO model (replace 'your_model.pt' with the path to your model)
model = YOLO('src/last.pt')  # Make sure to replace this with the correct path to your model

# Start capturing frames
with mss() as sct:
    while True:
        # Capture the frame
        frame = np.array(sct.grab(bbox))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR

        # Perform object detection using the YOLO model
        results = model(frame)

        # Check if there are detections and draw the results on the frame
        if len(results) > 0:
            for result in results[0].boxes.xyxy:  # Accessing the xyxy coordinates
                x1, y1, x2, y2 = result.tolist()  # Convert tensor to list
                conf = result.conf[0].item()  # Get confidence
                cls = result.cls[0].item()  # Get class ID

                if conf > 0.5:  # Threshold for detection
                    # Draw the bounding box and label
                    print("found 1")
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"{model.names[int(cls)]}: {conf:.2f}", 
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the processed frame
        cv2.imshow("Scrcpy Capture - Object Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
