import cv2
import numpy as np
import mss
import pygetwindow as gw

# Find the scrcpy window (for cropping)
window = gw.getWindowsWithTitle('SM-A356B')[0]

# Define the lower and upper bounds for the green color in HSV
lower_bound = np.array([40, 50, 50])  # lower bound for green (in HSV format)
upper_bound = np.array([80, 255, 255])  # upper bound for green (in HSV format)

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

        # Convert the frame to HSV (Hue, Saturation, Value) color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the green color using the specified color range
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours around detected power cubes
        for contour in contours:
            # Only consider large enough contours to avoid noise
            if cv2.contourArea(contour) > 100:
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Draw a rectangle around the power cube
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with detected power cubes
        cv2.imshow("Power Cube Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
