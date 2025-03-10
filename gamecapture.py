import cv2
import numpy as np
import mss
import pygetwindow as gw

# Find the scrcpy window (for cropping)
window = gw.getWindowsWithTitle('SM-A356B')[0]

# Fine-tune the lower and upper bounds for the green power cube color in HSV
lower_green = np.array([80, 100, 100])  # Lower bound for the specific green
upper_green = np.array([100, 255, 255])  # Upper bound for the specific green

# Fine-tune the minimum contour area and aspect ratio thresholds
min_area = 200  # Increase if small objects are being detected
min_aspect_ratio = 0.7  # Allow slight rectangle for power cubes
max_aspect_ratio = 1.3  # Allow slight rectangle for power cubes

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

        # Create a mask for detecting the green power cubes
        mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Find contours for power cubes
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_green:
            # Get contour area and filter based on size
            if cv2.contourArea(contour) > min_area:
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)  # Calculate aspect ratio

                # Filter based on aspect ratio to find square-like shapes (for power cubes)
                if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    # Draw a rectangle around the detected power cube
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Display the frame with detected power cubes
        cv2.imshow("Power Cube Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
