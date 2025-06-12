# imports type shit 
from movement import movement
from ppadb.client import Client as AdbClient
import time
import random

# Connect to ADB
client = AdbClient(host="127.0.0.1", port=5037)
devices = client.devices()

if len(devices) == 0:
    print("No devices connected.")
    exit()

device = devices[0]
print(f"Connected to {device.serial}")

# Simulate joystick movement
for x in range(1000):
    randomnum = random.randint(1, 8)
    start_x, start_y, end_x, end_y = movement(randomnum)
    device.shell(f"input swipe {start_x} {start_y} {end_x} {end_y} 3000")
    
