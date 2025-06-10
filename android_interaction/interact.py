#imports type shit 
from ppadb.client import Client as AdbClient
import time

# Connect to ADB
client = AdbClient(host="127.0.0.1", port=5037)
devices = client.devices()

if len(devices) == 0:
    print("No devices connected.")
    exit()

device = devices[0]
print(f"Connected to {device.serial}")

# Joystick center
movestick_x, movestick_y = 350, 825

# Directions (dx, dy offset) moving
directions = {
    "up": (0, -200),
    "down": (0, 200),
    "left": (-200, 0),
    "right": (200, 0),
    "up_right": (150, -150),
    "down_left": (-150, 150)
}


dx, dy = directions["up_right"]
end_x = movestick_x + dx
end_y = movestick_y + dy

# Simulate joystick
device.shell(f"input swipe {movestick_x} {movestick_y} {end_x} {end_y} 3000")
time.sleep(1)
