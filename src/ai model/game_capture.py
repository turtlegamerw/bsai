import cv2
import mss
import numpy as np
import win32gui
import os
import time

# SETTINGS
SAVE_DIR = "brawl_images"
INTERVAL = 0.5  # seconds between captures
WINDOW_NAME = "scrcpy"

os.makedirs(SAVE_DIR, exist_ok=True)

def get_window_rect(window_name):
    import win32gui
    windows = []
    win32gui.EnumWindows(
        lambda hwnd, wins: wins.append(hwnd) if window_name.lower() in win32gui.GetWindowText(hwnd).lower() else None,
        windows
    )
    if windows:
        return win32gui.GetWindowRect(windows[0])
    else:
        raise Exception(f"Window '{window_name}' not found.")

LEFT, TOP, RIGHT, BOTTOM = get_window_rect(WINDOW_NAME)
WIDTH, HEIGHT = RIGHT - LEFT, BOTTOM - TOP

count = 0
with mss.mss() as sct:
    while True:
        screenshot = sct.grab({"left": LEFT, "top": TOP, "width": WIDTH, "height": HEIGHT})
        frame = np.array(screenshot)

        filename = os.path.join(SAVE_DIR, f"frame_{count:04d}.png")
        cv2.imwrite(filename, frame)

        count += 1
        print(f"Captured: {filename}")

        time.sleep(INTERVAL)
