import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('unet_model.h5')

# Load a new image for inference
image_path = 'path/to/new_image.png'
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (256, 256)) / 255.0  # Normalize
image_input = np.expand_dims(image_resized, axis=0)  # Add batch dimension

# Perform prediction
predicted_mask = model.predict(image_input)

# Convert to binary mask (0 or 1)
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

# Visualize the result
plt.imshow(predicted_mask[0], cmap='gray')
plt.show()

# Optionally, overlay the result on the original image
result = cv2.addWeighted(image, 0.7, predicted_mask[0] * 255, 0.3, 0)
cv2.imshow("Segmented Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
