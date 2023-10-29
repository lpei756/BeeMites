import tensorflow as tf
import os

# Constants
CNN_MODEL_PATH = 'models/cnn_model.h5'
MOBILENET_MODEL_PATH = 'models/mobilenet_model.h5'
VALIDATION_DATA_PATH = 'images/validation'  # Assuming validation data is in 'images/validation'

# Load the models
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)

# Load the validation data
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DATA_PATH,
    target_size=(128, 128),  # Adjust based on your model input size
    batch_size=32,
    class_mode='binary'  # Adjust based on your classification type
)

# Evaluate the models
cnn_results = cnn_model.evaluate(validation_generator)
mobilenet_results = mobilenet_model.evaluate(validation_generator)

# Print the results
print("\nCNN Model Validation:")
print("Loss:", cnn_results[0])
print("Accuracy:", cnn_results[1])

print("\nMobileNet Model Validation:")
print("Loss:", mobilenet_results[0])
print("Accuracy:", mobilenet_results[1])
