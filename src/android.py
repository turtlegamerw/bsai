from ppadb.client import Client as AdbClient

# Connect to the ADB server and find the device
client = AdbClient(host="127.0.0.1", port=5037)
devices = client.devices()
if len(devices) == 0:
    print("No devices connected")
    exit()

device = devices[0]

# Function to send text input to the Android device
def send_text_to_android(text):
    # Replace spaces with %s for ADB shell command compatibility
    device.shell(f'input text "{text.replace(" ", "%s")}"')

# Get user input from the PC terminal
user_input = input("enter text: ")
send_text_to_android(user_input)
print("Text sent to Android device.")
