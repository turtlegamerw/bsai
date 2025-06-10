from ppadb.client import Client as AdbClient
import time


# Default is 127.0.0.1 and port 5037
client = AdbClient(host="127.0.0.1", port=5037)

# Get list of devices
devices = client.devices()

if len(devices) == 0:
    print("No devices connected.")
    exit()

device = devices[0]
print(f"Connected to {device.serial}")

#move 350 825 for walk
#tap 1500 950 for hyper
#move 1660 800 for super
#tap 1850 950 for gadget
#move 1960 700 for attack


x = 350
y = 825
device.shell(f"input tap {x} {y}")
time.sleep(5)
x = 1500
y = 950
device.shell(f"input tap {x} {y}")
time.sleep(5)
x = 1660
y = 800
device.shell(f"input tap {x} {y}")
time.sleep(5)
x = 1850
y = 950
device.shell(f"input tap {x} {y}")
time.sleep(5)
x = 1960
y = 700
device.shell(f"input tap {x} {y}")
exit()