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

# Start capturing frames
with mss() as sct:
    while True:
        # Capture the frame
        frame = np.array(sct.grab(bbox))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR

        # Process the frame (example: convert to grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the processed frame
        cv2.imshow("Scrcpy Capture - Grayscale", gray_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
