import tensorflow as tf
import os

# Constants
TRAINING_DATA_PATH = 'C:\\My Course\\2023S2\\BeeMitesTensorFlow\\images\\varroa mite'  # Path to training data
VALIDATION_DATA_PATH = 'C:\\My Course\\2023S2\\BeeMitesTensorFlow\\images\\validation mite'  # Path to validation data
MODEL_SAVE_PATH = 'models/mite_mobilenet_model.h5'

# Preprocess and load the training and validation data
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

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load the training and validation data with 'categorical' class_mode
train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important: Do not shuffle the data in validation generator
)

# Load MobileNetV2 with weights pre-trained on ImageNet, exclude the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Build and compile the model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # You might want to adjust this
    verbose=1
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
