import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 🔹 Define dataset path (Replace with your actual dataset path)
DATA_PATH = "location to your data brov"

# 🔹 Hyperparameters
BATCH_SIZE = 32
IMG_HEIGHT = 72
IMG_WIDTH = 128
EPOCHS = 30  # Increased epochs
DROPOUT_RATE_1 = 0.6
DROPOUT_RATE_2 = 0.4
L2_REG = 0.01  # L2 regularization factor

# 🔹 Load dataset with train-validation split
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=1234,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# 🔹 Get class names
CLASS_NAMES = train_dataset.class_names
NUM_CLASSES = len(CLASS_NAMES)

# 🔹 Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),  # Added contrast variation
    layers.RandomBrightness(0.2)  # Added brightness changes
])

# 🔹 Model architecture with L2 Regularization
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Data Augmentation
    data_augmentation,
    
    # First Convolutional Block
    layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                  kernel_regularizer=regularizers.l2(L2_REG)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Second Convolutional Block
    layers.Conv2D(64, (3,3), activation='relu', padding='same', 
                  kernel_regularizer=regularizers.l2(L2_REG)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Third Convolutional Block
    layers.Conv2D(128, (3,3), activation='relu', padding='same', 
                  kernel_regularizer=regularizers.l2(L2_REG)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Flatten Layer
    layers.Flatten(),
    
    # Fully Connected Layers with Increased Dropout
    layers.Dropout(DROPOUT_RATE_1),  
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(DROPOUT_RATE_2),
    layers.Dense(NUM_CLASSES, activation='softmax')  # Softmax for multi-class classification
])

# 🔹 Compile Model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# 🔹 Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

# 🔹 Model Callbacks (Fixed ModelCheckpoint Issue)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True),  # FIXED
    lr_scheduler
]

# 🔹 Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 🔹 Model summary
model.summary()

# 🔹 Plot Accuracy & Loss Graphs
epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(16, 5))
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.legend(loc='lower right')
plt.title('Accuracy vs Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()

plt.figure(figsize=(16, 5))
plt.plot(epochs_range, history.history['loss'], label='Training Loss', color='green')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss', color='red')
plt.legend(loc='upper right')
plt.title('Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

# 🔹 Single Image Prediction from Validation Set
plt.figure(figsize=(6, 3))  
for images, labels in val_dataset.take(1):
    sample_image = images[1]
    true_label = labels[1]

    sample_image = tf.expand_dims(sample_image, axis=0)

    predictions = model.predict(sample_image)
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class = CLASS_NAMES[predicted_class_index]

    plt.imshow(sample_image[0].numpy().astype("uint8"))
    plt.title(f"True label: {CLASS_NAMES[true_label.numpy()]}\nPredicted label: {predicted_class}")
    plt.axis('off')

plt.show()
