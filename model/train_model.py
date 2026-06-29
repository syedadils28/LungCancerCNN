"""
Train Random Forest and KNN classifiers on CNN-extracted features (transfer
learning) instead of raw pixels. The CNN ('cancer.h5') is used purely as a
feature extractor here -- its second-to-last layer (256-dim) becomes the
input vector for RF/KNN.

Run from the model/ folder (which contains train/, test/, valid/, cancer.h5).

Outputs (saved in model/):
    rf_model.pkl
    knn_model.pkl
    rf_knn_comparison.png

Usage:
    cd model/
    python train_model.py
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimg

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR    = os.path.join(BASE, 'train')
TEST_DIR     = os.path.join(BASE, 'test')
SAVE_DIR     = BASE
CNN_PATH     = os.path.join(BASE, 'cancer.h5')
CNN_IMG_SIZE = (150, 150)

CLASS_NAMES = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma',
]


# ── Feature Extractor ─────────────────────────────────────────────────────────
def build_feature_extractor():
    """
    Load the CNN and strip the final softmax layer so it outputs
    the 256-dim penultimate feature vector instead of class probabilities.
    Works with Sequential models (no cnn.input needed).
    """
    print(f"  Loading CNN from: {CNN_PATH}")
    cnn = load_model(CNN_PATH)

    # Warm-up pass so Keras initialises the Sequential graph
    dummy = np.zeros((1, *CNN_IMG_SIZE, 3), dtype=np.float32)
    cnn.predict(dummy, verbose=0)

    # Build extractor = all layers except the last softmax
    extractor = tf.keras.Sequential(cnn.layers[:-1])
    extractor.build((None, *CNN_IMG_SIZE, 3))

    print(f"  Extractor output shape: {extractor.output_shape}")
    return extractor


# ── Image Loading + Feature Extraction ───────────────────────────────────────
def load_split_as_features(folder, extractor, batch_size=32):
    paths, labels = [], []

    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] missing folder: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    if not paths:
        return np.array([]), np.array([])

    features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_imgs  = []
        for p in batch_paths:
            try:
                img = kimg.load_img(p, target_size=CNN_IMG_SIZE)
                batch_imgs.append(kimg.img_to_array(img) / 255.0)
            except Exception as e:
                print(f"  [SKIP] {p}: {e}")
                batch_imgs.append(np.zeros((*CNN_IMG_SIZE, 3)))
        feats = extractor.predict(np.array(batch_imgs), verbose=0)
        features.append(feats)
        print(f"  extracted {min(i + batch_size, len(paths))}/{len(paths)}")

    return np.vstack(features), np.array(labels)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RF + KNN Training (CNN-feature based)")
    print("=" * 55)
    print(f"\nCNN  : {CNN_PATH}")
    print(f"Train: {TRAIN_DIR}")
    print(f"Test : {TEST_DIR}")
    print(f"Save : {SAVE_DIR}\n")

    print("[1] Building feature extractor...")
    extractor = build_feature_extractor()

    print("\n[2] Extracting train features...")
    X_train, y_train = load_split_as_features(TRAIN_DIR, extractor)
    print(f"    Train samples: {len(X_train)}")

    print("\n[3] Extracting test features...")
    X_test, y_test = load_split_as_features(TEST_DIR, extractor)
    print(f"    Test samples : {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit(
            "\n[ERROR] No images found.\n"
            "Check that train/ and test/ exist inside model/ with 4 class subfolders."
        )

    print("\n[4] Training Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid={'n_estimators': [200, 400], 'max_depth': [None, 30]},
        cv=3, n_jobs=-1, verbose=1,
    )
    rf_grid.fit(X_train, y_train)
    rf      = rf_grid.best_estimator_
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)
    print(f"\n    Best RF params : {rf_grid.best_params_}")
    print(f"    RF Test Accuracy: {rf_acc * 100:.2f}%")
    print(classification_report(y_test, rf_pred, target_names=CLASS_NAMES))

    print("\n[5] Training KNN...")
    knn_grid = GridSearchCV(
        KNeighborsClassifier(weights='distance', n_jobs=-1),
        param_grid={'n_neighbors': [3, 5, 7, 9]},
        cv=3, n_jobs=-1, verbose=1,
    )
    # preprocessing: KNN is sensitive to feature scaling, but CNN features are already normalized.
    knn_grid.fit(X_train, y_train)
    knn      = knn_grid.best_estimator_
    knn_pred = knn.predict(X_test)
    knn_acc  = accuracy_score(y_test, knn_pred)
    print(f"\n    Best KNN params : {knn_grid.best_params_}")
    print(f"    KNN Test Accuracy: {knn_acc * 100:.2f}%")
    print(classification_report(y_test, knn_pred, target_names=CLASS_NAMES))

    rf_path  = os.path.join(SAVE_DIR, 'rf_model.pkl')
    knn_path = os.path.join(SAVE_DIR, 'knn_model.pkl')
    joblib.dump(rf,  rf_path)
    joblib.dump(knn, knn_path)
    print(f"\n[6] Saved: {os.path.normpath(rf_path)}")
    print(f"    Saved: {os.path.normpath(knn_path)}")

    plt.figure(figsize=(6, 4))
    plt.bar(
        ['Random Forest', 'KNN'],
        [rf_acc * 100, knn_acc * 100],
        color=['seagreen', 'darkorange']
    )
    plt.ylabel('Test Accuracy (%)')
    plt.title('RF vs KNN Test Accuracy (CNN-feature based)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plot_path = os.path.join(BASE, 'rf_knn_comparison.png')
    plt.savefig(plot_path)
    print(f"    Plot : {os.path.normpath(plot_path)}")

    print("\n" + "=" * 55)
    print("  Done! model/ now contains:")
    print("    cancer.h5  rf_model.pkl  knn_model.pkl")
    print("=" * 55)


if __name__ == '__main__':
    main()