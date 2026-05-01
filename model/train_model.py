# ============================================================
# Lung Cancer CNN - Model Training Script
# Run this inside the /model/ folder with: python train_model.py
# OR open model.ipynb in Jupyter and run cell by cell
# ============================================================

# Cell 1: Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

print("TensorFlow version:", tf.__version__)

# ── Cell 2: Configuration ────────────────────────────────────
TRAIN_DIR = 'train/'
VALID_DIR = 'valid/'
TEST_DIR  = 'test/'
IMG_SIZE  = (150, 150)
BATCH     = 32
EPOCHS    = 20

# ── Cell 3: Data Augmentation ────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_gen = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical'
)
valid_data = valid_gen.flow_from_directory(
    VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical'
)
test_data = test_gen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical', shuffle=False
)

print("Classes:", train_data.class_indices)
print("Train samples:", train_data.samples)
print("Valid samples:", valid_data.samples)

# ── Cell 4: Build CNN Model ───────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    # Block 4
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Classifier Head
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')   # 4 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Cell 5: Callbacks ─────────────────────────────────────────
callbacks = [
    ModelCheckpoint('../cancer.h5', save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# ── Cell 6: Train ─────────────────────────────────────────────
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=valid_data,
    callbacks=callbacks
)

print("\n✅ Model saved as cancer.h5")

# ── Cell 7: Evaluate on Test Set ──────────────────────────────
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Test Loss:     {loss:.4f}")

# ── Cell 8: Plot Results ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='orangered')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train Loss', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='orangered')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
print("Plot saved as training_results.png")
