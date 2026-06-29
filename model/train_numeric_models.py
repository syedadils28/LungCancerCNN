"""
train_numeric_models.py
========================
Trains a Random Forest and a KNN classifier on the 15-feature
questionnaire (age, smoking, wheezing, etc.) to predict the SAME
4 classes used by the CT-scan CNN pipeline:

    0 = adenocarcinoma
    1 = large.cell.carcinoma
    2 = normal
    3 = squamous.cell.carcinoma

Usage:
    cd model/
    python train_numeric_models.py

Outputs (saved in model/):
    numeric_rf_model.pkl
    numeric_knn_model.pkl
    numeric_scaler.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Save to the same folder as this script (model/) ──────────────────────────
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
N = 6000

# ── Generate symptom features ─────────────────────────────────────────────────
age                   = np.random.randint(30, 85, N)
gender                = np.random.choice([0, 1], N)
smoking               = np.random.choice([0, 1], N, p=[0.4, 0.6])
yellow_fingers        = (smoking & (np.random.rand(N) > 0.4)).astype(int)
anxiety               = np.random.choice([0, 1], N, p=[0.6, 0.4])
peer_pressure         = np.random.choice([0, 1], N, p=[0.6, 0.4])
chronic_disease       = np.random.choice([0, 1], N, p=[0.7, 0.3])
fatigue               = np.random.choice([0, 1], N, p=[0.5, 0.5])
allergy               = np.random.choice([0, 1], N, p=[0.6, 0.4])
wheezing              = (smoking & (np.random.rand(N) > 0.5)).astype(int)
alcohol               = np.random.choice([0, 1], N, p=[0.55, 0.45])
coughing              = np.random.choice([0, 1], N, p=[0.5, 0.5])
shortness_of_breath   = np.random.choice([0, 1], N, p=[0.55, 0.45])
swallowing_difficulty = np.random.choice([0, 1], N, p=[0.75, 0.25])
chest_pain            = np.random.choice([0, 1], N, p=[0.65, 0.35])

# ── Step 1: overall cancer risk ───────────────────────────────────────────────
risk = (
    0.35 * smoking + 0.15 * (age > 60).astype(int) +
    0.10 * chronic_disease + 0.08 * wheezing +
    0.07 * chest_pain + 0.06 * shortness_of_breath +
    0.05 * fatigue + 0.05 * coughing +
    0.04 * yellow_fingers + 0.03 * swallowing_difficulty +
    0.02 * alcohol + np.random.rand(N) * 0.10
)
has_cancer = (risk > 0.45).astype(int)

# ── Step 2: assign subtype if cancer ─────────────────────────────────────────
squamous_score = (2.0 * smoking + 1.5 * wheezing + 1.2 * chest_pain
                  + 0.5 * (age > 60).astype(int) + np.random.randn(N) * 0.4)
adeno_score    = (1.0 * shortness_of_breath + 0.8 * (age <= 60).astype(int)
                  + 0.6 * (1 - smoking) + 0.5 * allergy + np.random.randn(N) * 0.4)
large_score    = (1.5 * fatigue + 1.3 * chronic_disease + 1.0 * swallowing_difficulty
                  + 0.5 * coughing + np.random.randn(N) * 0.4)

subtype_idx = np.argmax(np.stack([adeno_score, large_score, squamous_score], axis=1), axis=1)

# CLASS_NAMES order: 0=adenocarcinoma, 1=large.cell.carcinoma, 2=normal, 3=squamous.cell.carcinoma
label = np.where(
    has_cancer == 0, 2,
    np.where(subtype_idx == 0, 0, np.where(subtype_idx == 1, 1, 3))
)

df = pd.DataFrame({
    'Age': age, 'Gender': gender, 'Smoking': smoking,
    'Yellow_Fingers': yellow_fingers, 'Anxiety': anxiety,
    'Peer_Pressure': peer_pressure, 'Chronic_Disease': chronic_disease,
    'Fatigue': fatigue, 'Allergy': allergy, 'Wheezing': wheezing,
    'Alcohol_Consuming': alcohol, 'Coughing': coughing,
    'Shortness_of_Breath': shortness_of_breath,
    'Swallowing_Difficulty': swallowing_difficulty, 'Chest_Pain': chest_pain,
    'Label': label
})

print(f"Dataset: {N} samples")
print(f"Class distribution:\n{df['Label'].value_counts().sort_index()}\n")

X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

CLASS_NAMES = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# ── Random Forest ─────────────────────────────────────────────────────────────
print("[1] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"    RF Test Accuracy: {rf_acc * 100:.2f}%")
print(classification_report(y_test, rf.predict(X_test), target_names=CLASS_NAMES, digits=3))

# ── KNN (needs scaling) ───────────────────────────────────────────────────────
print("[2] Training KNN...")
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1)
knn.fit(X_train_s, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test_s))
print(f"    KNN Test Accuracy: {knn_acc * 100:.2f}%")
print(classification_report(y_test, knn.predict(X_test_s), target_names=CLASS_NAMES, digits=3))

# ── Save ──────────────────────────────────────────────────────────────────────
rf_path     = os.path.join(SAVE_DIR, 'numeric_rf_model.pkl')
knn_path    = os.path.join(SAVE_DIR, 'numeric_knn_model.pkl')
scaler_path = os.path.join(SAVE_DIR, 'numeric_scaler.pkl')

joblib.dump(rf,     rf_path)
joblib.dump(knn,    knn_path)
joblib.dump(scaler, scaler_path)

print(f"\n[3] Saved:")
print(f"    {os.path.normpath(rf_path)}")
print(f"    {os.path.normpath(knn_path)}")
print(f"    {os.path.normpath(scaler_path)}")
print("\nDone! Now run app.py")