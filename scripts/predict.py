import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("models/skin_disease_model.h5")

# Define image classes (same as training classes)
CLASS_NAMES = ['Eczema', 'Melanoma', 'Atopic Dermatitis', 'BCC', 'NV',
               'BKL', 'Psoriasis', 'Seborrheic Keratoses', 'Fungal Infections', 'Warts']

# Function to predict an image
def predict_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image)
    class_index = np.argmax(prediction)  # Get highest probability class
    confidence = np.max(prediction) * 100  # Get confidence percentage

    print(f"✅ Predicted Disease: {CLASS_NAMES[class_index]} with {confidence:.2f}% confidence.")

# Test the function with an image
image_path = "test_image/sample.jpg"  # Change to your test image path
predict_image(image_path)
