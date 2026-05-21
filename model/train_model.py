import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)

BASE      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE, 'train')
VALID_DIR = os.path.join(BASE, 'valid')
TEST_DIR  = os.path.join(BASE, 'test')
SAVE_PATH = os.path.join(BASE, '..', 'cancer.h5')

IMG_SIZE = (150, 150)
BATCH    = 32
EPOCHS   = 20

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_gen  = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
valid_data = val_gen.flow_from_directory(VALID_DIR,   target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
test_data  = test_gen.flow_from_directory(TEST_DIR,   target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical', shuffle=False)

print("Classes:", train_data.class_indices)
print("Train samples:", train_data.samples)
print("Valid samples:", valid_data.samples)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(150,150,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("\nTraining started...")
history = model.fit(train_data, epochs=EPOCHS, validation_data=valid_data, callbacks=callbacks)

loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Test Loss:     {loss:.4f}")
print(f"\nModel saved to: {os.path.normpath(SAVE_PATH)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='orangered')
axes[0].set_title('Accuracy'); axes[0].legend()
axes[1].plot(history.history['loss'],     label='Train', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val',   color='orangered')
axes[1].set_title('Loss'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'training_results.png'))
print("Plot saved.")