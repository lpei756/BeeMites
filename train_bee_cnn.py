import tensorflow as tf
import os

# Constants
BASE_DATA_PATH = 'C:\\My Course\\2023S2\\BeeMitesTensorFlow\\images\\bee'
MODEL_SAVE_PATH = 'models/bee_cnn_model.h5'

# Preprocess and load the training data for bees
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    BASE_DATA_PATH,  # Pointing directly to the 'bee' directory
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    verbose=1
)

# Save the trained model
model.save(MODEL_SAVE_PATH, save_format='h5')  # Explicitly set save format to avoid the warning
print(f"Model saved to {MODEL_SAVE_PATH}")
