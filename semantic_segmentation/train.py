import tensorflow as tf
from data_preprocessing import load_data
from unet_model import unet
from sklearn.model_selection import train_test_split

# Load data
image_dir = 'path/to/images'
mask_dir = 'path/to/masks'
images, masks = load_data(image_dir, mask_dir)

# Split data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Initialize U-Net model
model = unet(input_size=(256, 256, 3))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=4)

# Save the trained model
model.save('unet_model.h5')
