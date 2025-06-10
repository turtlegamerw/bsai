from ppadb.client import Client as AdbClient

# Default is 127.0.0.1 and port 5037
client = AdbClient(host="127.0.0.1", port=5037)

# Get list of devices
devices = client.devices()

if len(devices) == 0:
    print("No devices connected.")
    exit()

device = devices[0]
print(f"Connected to {device.serial}")

# Run a shell command
print(device.shell("echo Hello from your phone!"))
