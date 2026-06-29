"""
Evaluate the CNN, Random Forest, and KNN models on the real test set and
write the results to metrics.json. The Flask app's /performance route reads
this file and renders real numbers instead of hardcoded placeholders.

Run this AFTER both:
    1. the CNN training script (produces ../cancer.h5)
    2. train_rf_knn.py            (produces ../rf_model.pkl, ../knn_model.pkl)

Usage (from the `dataset` folder, same layout as train_rf_knn.py):
    python evaluate_models.py

Output:
    ../metrics.json
"""

import os
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as kimg

BASE      = os.path.dirname(os.path.abspath(__file__))
TEST_DIR  = os.path.join(BASE, 'test')
SAVE_DIR  = BASE   # cancer.h5 / rf_model.pkl / knn_model.pkl all live alongside this script
CNN_PATH  = os.path.join(SAVE_DIR, 'cancer.h5')
RF_PATH   = os.path.join(SAVE_DIR, 'rf_model.pkl')
KNN_PATH  = os.path.join(SAVE_DIR, 'knn_model.pkl')
OUT_PATH  = os.path.join(SAVE_DIR, 'metrics.json')

CNN_IMG_SIZE = (150, 150)

CLASS_NAMES = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma',
]
DISPLAY_NAMES = {
    'adenocarcinoma':          'Adenocarcinoma',
    'large.cell.carcinoma':    'Large Cell Carcinoma',
    'normal':                  'Normal',
    'squamous.cell.carcinoma': 'Squamous Cell Carcinoma',
}


def load_test_images():
    """Returns (image_arrays [N,150,150,3], labels [N])."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] missing folder: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            paths.append(os.path.join(cls_dir, fname))
            labels.append(idx)

    imgs = []
    for p in paths:
        img = kimg.load_img(p, target_size=CNN_IMG_SIZE)
        imgs.append(kimg.img_to_array(img) / 255.0)
    return np.array(imgs), np.array(labels), paths


def block_for(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    per_class = {
        DISPLAY_NAMES[CLASS_NAMES[i]]: {
            'precision': round(float(p_c[i]) * 100, 2),
            'recall':    round(float(r_c[i]) * 100, 2),
            'f1':        round(float(f_c[i]) * 100, 2),
        }
        for i in range(len(CLASS_NAMES))
    }
    return {
        'accuracy':  round(float(acc) * 100, 2),
        'precision': round(float(precision) * 100, 2),
        'recall':    round(float(recall) * 100, 2),
        'f1':        round(float(f1) * 100, 2),
        'per_class': per_class,
        'n_test_samples': int(len(y_true)),
    }


def main():
    if not os.path.exists(CNN_PATH):
        raise SystemExit(f"CNN model not found at {CNN_PATH}. Train it first.")
    if not os.path.exists(RF_PATH) or not os.path.exists(KNN_PATH):
        raise SystemExit("rf_model.pkl / knn_model.pkl not found. Run train_rf_knn.py first.")

    print("Loading models...")
    cnn = load_model(CNN_PATH)
    # A Sequential model loaded from .h5 hasn't been "called" yet, so
    # Model(inputs=cnn.input, ...) raises AttributeError. Call it once on a
    # dummy batch to build it, then construct the extractor as a Sequential
    # of all layers except the last (matches app.py's get_feature_extractor).
    dummy = np.zeros((1, *CNN_IMG_SIZE, 3), dtype=np.float32)
    cnn.predict(dummy, verbose=0)
    extractor = tf.keras.Sequential(cnn.layers[:-1])
    extractor.build((None, *CNN_IMG_SIZE, 3))
    rf = joblib.load(RF_PATH)
    knn = joblib.load(KNN_PATH)

    print("Loading test images...")
    X_img, y_true, paths = load_test_images()
    print("Test samples:", len(X_img))
    if len(X_img) == 0:
        raise SystemExit("No test images found. Check TEST_DIR / folder names.")

    print("Running CNN predictions...")
    cnn_proba = cnn.predict(X_img, verbose=0)
    cnn_pred = np.argmax(cnn_proba, axis=1)

    print("Extracting CNN features for RF/KNN...")
    feats = extractor.predict(X_img, verbose=0)

    print("Running RF predictions...")
    rf_pred = rf.predict(feats)

    print("Running KNN predictions...")
    knn_pred = knn.predict(feats)

    print("\nComputing metrics...")
    metrics = {
        'cnn': block_for(y_true, cnn_pred),
        'rf':  block_for(y_true, rf_pred),
        'knn': block_for(y_true, knn_pred),
    }

    agreement = {
        'cnn_rf':  round(float(np.mean(cnn_pred == rf_pred)) * 100, 2),
        'cnn_knn': round(float(np.mean(cnn_pred == knn_pred)) * 100, 2),
        'rf_knn':  round(float(np.mean(rf_pred == knn_pred)) * 100, 2),
    }
    metrics['agreement'] = agreement

    # True vs predicted counts per class (for the distribution chart)
    distribution = {}
    for i, cls in enumerate(CLASS_NAMES):
        name = DISPLAY_NAMES[cls]
        distribution[name] = {
            'true':      int(np.sum(y_true == i)),
            'cnn_pred':  int(np.sum(cnn_pred == i)),
            'rf_pred':   int(np.sum(rf_pred == i)),
            'knn_pred':  int(np.sum(knn_pred == i)),
        }
    metrics['distribution'] = distribution
    metrics['generated_at'] = datetime.now().isoformat()
    metrics['class_names'] = [DISPLAY_NAMES[c] for c in CLASS_NAMES]

    with open(OUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved: {os.path.normpath(OUT_PATH)}")
    print(f"\nCNN  accuracy: {metrics['cnn']['accuracy']}%")
    print(f"RF   accuracy: {metrics['rf']['accuracy']}%")
    print(f"KNN  accuracy: {metrics['knn']['accuracy']}%")
    print(f"\nAgreement -- CNN/RF: {agreement['cnn_rf']}%  CNN/KNN: {agreement['cnn_knn']}%  RF/KNN: {agreement['rf_knn']}%")


if __name__ == '__main__':
    main()