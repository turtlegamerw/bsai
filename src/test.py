import cv2
import numpy as np
import pygetwindow as gw
from mss import mss

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

# Load a pre-trained Haar Cascade for object detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start capturing frames
with mss() as sct:
    while True:
        # Capture the frame
        frame = np.array(sct.grab(bbox))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR

        # Detect objects (e.g., faces)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the processed frame
        cv2.imshow("Scrcpy Capture - Object Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            exit()

cv2.destroyAllWindows()
