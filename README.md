# 🫁 LungVision AI — Lung Cancer Detection & Classification System

An end-to-end machine learning web application that detects and classifies lung cancer from CT scan images into 4 distinct categories. It features a Convolutional Neural Network (CNN) as the primary diagnostic model, with Random Forest and KNN classifiers trained on CNN-extracted features for cross-model validation — plus a separate symptom-questionnaire based risk predictor. Built with Flask, Chart.js, and a clean glassmorphism-inspired dashboard.

---

## 🌟 Key Features

- **Deep Learning Classification**: CNN trained on chest CT scan images, predicting 4 lung tissue classes with full confidence/probability breakdown.
- **Multi-Model Validation**: Random Forest and KNN classifiers trained on the CNN's 256-dim extracted features, shown alongside the CNN result for model agreement analysis.
- **Symptom-Based Risk Predictor**: A separate numeric predictor (`/predict_numeric`) that classifies risk from a 15-question patient symptom questionnaire (age, smoking, wheezing, chest pain, etc.) using RF and KNN — no CT scan required.
- **Model Comparison & Agreement Dashboard**: Visual verdict cards showing when CNN, RF, and KNN agree or disagree, per-prediction.
- **Performance Analytics Page**: Real accuracy/precision/recall/F1 metrics, per-class F1 tables, agreement percentages, and a multi-model radar chart — all pulled live from `metrics.json`.
- **Secure Authentication**: Session-based login protecting all diagnostic routes.
- **Demo Mode**: If `cancer.h5` isn't present, the app falls back to randomized predictions so the UI can still be tested end-to-end.

---

## 📂 Project Directory Structure

```text
LungCancerCNN/
├── model/
│   ├── train_cnn.py             # Trains the CNN on CT scan images → cancer.h5
│   ├── train_model.py           # Trains RF & KNN on CNN-extracted features
│   ├── train_numeric_models.py  # Trains RF & KNN on the 15-feature symptom questionnaire
│   ├── evaluate_models.py       # Generates metrics.json (accuracy/precision/recall/F1)
│   ├── metrics.json             # Evaluation results consumed by performance.html
│   ├── cancer.h5                # Trained CNN weights (gitignored)
│   ├── rf_model.pkl             # Random Forest — CT scan features (gitignored)
│   ├── knn_model.pkl            # KNN — CT scan features (gitignored)
│   ├── numeric_rf_model.pkl     # Random Forest — symptom questionnaire (gitignored)
│   ├── numeric_knn_model.pkl    # KNN — symptom questionnaire (gitignored)
│   └── numeric_scaler.pkl       # StandardScaler for numeric features (gitignored)
├── static/
│   ├── style.css
│   └── predict_numeric.css
├── templates/
│   ├── base.html
│   ├── index.html               # Landing page
│   ├── login.html                # Authentication
│   ├── predict.html              # CT scan upload & CNN/RF/KNN prediction
│   ├── predict_numeric.html      # Symptom questionnaire risk predictor
│   ├── performance.html          # Model performance analytics dashboard
│   └── charts.html               # Dataset & training visualizations
├── upload/                       # Uploaded CT scan images (auto-created, gitignored)
├── app.py                        # Main Flask backend
├── requirements.txt
└── README.md
```

> **Note:** The dataset itself, trained model weights (`.h5`, `.pkl`), and uploaded images are excluded from this repository via `.gitignore` — see [Dataset Details](#-dataset-details) below for how to obtain the data.

---

## 🔬 Dataset Details

This project uses the **Chest CT-Scan Images Dataset** from Kaggle:

🔗 **[Chest CT-Scan Images (Kaggle)](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)**

The dataset contains chest CT scan images pre-split into `train/`, `valid/`, and `test/` folders, organized into 4 classes:

| # | Class | Description |
|---|-------|-------------|
| 1 | **Adenocarcinoma** | Most common subtype of non-small cell lung cancer (NSCLC); arises from glandular cells. |
| 2 | **Large Cell Carcinoma** | Poorly differentiated NSCLC subtype with large atypical cells. |
| 3 | **Normal** | No suspicious lung lesions detected. |
| 4 | **Squamous Cell Carcinoma** | NSCLC subtype typically linked to long-term smoking; often central/hilar masses. |

### Expected folder structure after download

```text
model/
├── train/
│   ├── adenocarcinoma/
│   ├── large.cell.carcinoma/
│   ├── normal/
│   └── squamous.cell.carcinoma/
├── valid/
│   └── (same 4 subfolders)
└── test/
    └── (same 4 subfolders)
```

Download the dataset from Kaggle, extract it, and place the `train/`, `valid/`, and `test/` folders inside the `model/` directory before running the training scripts.

---

## 🧠 Model Details

### 1. CNN — Primary Diagnostic Model
Trained end-to-end on raw CT scan pixels; learns spatial patterns invisible to the other models.

- **Input size:** 150×150 RGB
- **Architecture:** 4× (Conv2D → BatchNormalization → MaxPooling2D) blocks → Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)
- **Optimizer:** Adam (lr = 0.001), with `EarlyStopping` and `ReduceLROnPlateau` callbacks
- **Framework:** TensorFlow / Keras
- Saved as `model/cancer.h5`

### 2. Random Forest & KNN — Validation Models (CT scan)
Instead of training on raw pixels, `train_model.py` uses the CNN's own **256-dim penultimate layer** as a feature extractor (transfer learning). RF and KNN are then trained on these extracted features and used purely as a cross-check against the CNN's prediction.

- Random Forest: `GridSearchCV` over `n_estimators` and `max_depth`
- KNN: `GridSearchCV` over `n_neighbors`, distance-weighted

### 3. Random Forest & KNN — Symptom Questionnaire Model
A second, independent pipeline (`train_numeric_models.py`) trains RF and KNN on 15 patient-reported symptoms (age, gender, smoking, yellow fingers, anxiety, chronic disease, wheezing, chest pain, etc.) to predict the same 4 classes — enabling a screening flow that doesn't require a CT scan at all.

### 📈 Current Evaluation Snapshot (from `metrics.json`)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| 🧠 CNN | 35.88% | 26.82% | 35.88% | 19.91% |
| 🌲 Random Forest | 85.42% | 85.91% | 85.42% | 85.42% |
| 📍 KNN | 85.65% | 85.81% | 85.65% | 85.66% |

*(864 held-out test CT scans. Note the current CNN checkpoint is under-trained relative to RF/KNN — retraining `train_cnn.py` for more epochs is recommended before relying on it as the primary model.)*

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional but recommended) A GPU for CNN training

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd LungCancerCNN
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download the [Chest CT-Scan Images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle and place the `train/`, `valid/`, and `test/` folders inside `model/` (see [Dataset Details](#-dataset-details)).

---

## 🚀 Running the Project

### Phase 1: Train the models (required for real predictions)

```bash
cd model

# 1. Train the CNN on CT scan images → cancer.h5
python train_cnn.py

# 2. Train RF & KNN on CNN-extracted features → rf_model.pkl, knn_model.pkl
python train_model.py

# 3. Train RF & KNN on the symptom questionnaire → numeric_*.pkl
python train_numeric_models.py

# 4. Generate real evaluation metrics → metrics.json
python evaluate_models.py
```

> **Demo mode:** If `cancer.h5` is missing, the app still runs and returns randomized predictions for the CT-scan flow — useful for testing the UI without training first.

### Phase 2: Run the web app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🔐 Login Credentials

- **Username:** `admin`
- **Password:** `admin123`

To change credentials, set environment variables before running:
```bash
export APP_USER=youruser
export APP_PASS=yourpassword
export SECRET_KEY=your-random-secret-key
```

---

## 📊 Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page |
| Login | `/login` | Authentication |
| Predict | `/predict` | Upload a CT scan → CNN, RF & KNN predictions + agreement verdict |
| Predict (Numeric) | `/predict_numeric` | 15-question symptom form → RF & KNN risk classification |
| Performance | `/performance` | Model accuracy/precision/recall/F1, per-class breakdown, agreement %, radar chart |
| Charts | `/charts` | Dataset & training visualizations |
| Logout | `/logout` | End session |

---

## ⚠️ Disclaimer

This system is developed for **educational purposes only** as part of an MCA project. It is **not** a certified medical diagnostic tool. All predictions must be reviewed and confirmed by a qualified medical professional before any clinical decision is made.
