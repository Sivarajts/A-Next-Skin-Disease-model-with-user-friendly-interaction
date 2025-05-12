import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model("models/skin_disease_model.h5")

# Define test dataset path
TEST_DATASET_PATH = "data/IMG_CLASSES/"

# Image settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load test images
test_generator = test_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")
