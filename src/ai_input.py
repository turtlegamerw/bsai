import mss
import numpy as np
import cv2
from ultralytics import YOLO
import win32gui
from input.gadget import clickgadget
from ppadb.client import Client as AdbClient

#load in model
model = YOLO('src/ai model/best.pt')

# Connect to ADB
client = AdbClient(host="127.0.0.1", port=5037)
devices = client.devices()
if len(devices) == 0:
    print("No devices connected.")
    exit()
device = devices[0]
print(f"Connected to {device.serial}")

#getting windows size and stuff
def get_window_rect(window_name):
    windows = []
    def enum_windows(hwnd, lParam):
        if window_name.lower() in win32gui.GetWindowText(hwnd).lower():
            windows.append(hwnd)
    win32gui.EnumWindows(enum_windows, None)
    if windows:
        return win32gui.GetWindowRect(windows[0])
    else:
        raise Exception(f"Window '{window_name}' not found.")

window_name = "scrcpy" 
left, top, right, bottom = get_window_rect(window_name)
width = right - left
height = bottom - top

monitor = {
    "top": top,
    "left": left,
    "width": width,
    "height": height
}

#main loop
with mss.mss() as sct:
    while True:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        frame = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        results = model(frame)
        result = results[0]

        detected_classes = [model.names[int(box.cls.cpu().numpy()[0])] for box in result.boxes]
        detected_classes = list(set(detected_classes))

        if 'gadget_charged' in detected_classes:
            #clicks gadget
            x_gadget, y_gadget = clickgadget()
            device.shell(f"input tap {x_gadget} {y_gadget} ")

        
            

        annotated_frame = result.plot()
        cv2.imshow("YOLOv8 scrcpy Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()