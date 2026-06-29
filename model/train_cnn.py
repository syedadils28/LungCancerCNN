"""
Train the CNN on lung cancer CT images and save cancer.h5 inside model/.

Folder structure expected:
    model/
        train/
            adenocarcinoma/
            large.cell.carcinoma/
            normal/
            squamous.cell.carcinoma/
        valid/
            (same 4 subfolders)
        test/
            (same 4 subfolders)

Usage:
    cd model/
    python train_cnn.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))   # → model/
TRAIN_DIR = os.path.join(BASE, 'train')
VALID_DIR = os.path.join(BASE, 'valid')
TEST_DIR  = os.path.join(BASE, 'test')
SAVE_PATH = os.path.join(BASE, 'cancer.h5')             # → model/cancer.h5

IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
EPOCHS     = 30
NUM_CLASSES = 4

# ── Data Generators ───────────────────────────────────────────────────────────
# Normalization. ImageDataGenerator(rescale=1./255) divides every pixel
# value (which ranges 0–255) down into the 0–1 range. Neural networks train
# far more reliably on small, consistent numbers than on raw 0–255 integers.

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
val_gen  = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)
# Resizing. Every image — regardless of its original size — gets resized
# to 150×150 pixels (IMG_SIZE = (150, 150)). The CNN's input layer expects
# a fixed shape, so this step forces consistency across the whole dataset.
train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)
val_data = val_gen.flow_from_directory(
    VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_data = test_gen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

print("\nClass indices:", train_data.class_indices)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),   # ← penultimate layer (RF/KNN feature vector)
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n[Training CNN...]\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(test_data, verbose=0)
print(f"\nTest Accuracy : {acc*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(SAVE_PATH)
print(f"\n[Saved] cancer.h5 → {SAVE_PATH}")

# ── Training plot ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train Acc')
ax1.plot(history.history['val_accuracy'], label='Val Acc')
ax1.set_title('Accuracy'); ax1.legend()

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss'); ax2.legend()

plot_path = os.path.join(BASE, 'training_results.png')
plt.tight_layout()
plt.savefig(plot_path)
print(f"[Saved] training_results.png → {plot_path}")
print("\nDone! Now run: python train_model.py")