#cool stuff
from super import aimsuper
from movement import movement
from hyper import clickhyper
from gadget import clickgadget
from aiming import aim
# imports type shit 
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

#clicks hyper
x_hyper, y_hyper = clickhyper()
device.shell(f"input tap {x_hyper} {y_hyper} ")

#clicks gadget
x_gadget, y_gadget = clickgadget()
device.shell(f"input tap {x_gadget} {y_gadget} ")

#moves in a random direction for now
randomnum = random.randint(1, 8)
start_x, start_y, end_x, end_y = movement(randomnum)
device.shell(f"input swipe {start_x} {start_y} {end_x} {end_y} 3000")

#aimes in a random direction for now
randomnum = random.randint(1, 8)
start_x, start_y, end_x, end_y = aim(randomnum)
device.shell(f"input swipe {start_x} {start_y} {end_x} {end_y} 3000")

#uses super in a random direction for now
randomnum = random.randint(1, 8)
start_x, start_y, end_x, end_y = aimsuper(randomnum)
device.shell(f"input swipe {start_x} {start_y} {end_x} {end_y} 3000")
