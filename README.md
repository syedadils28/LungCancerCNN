# 🫁 LungVision AI — Lung Cancer Detection System

Advanced AI platform for early detection and classification of lung cancer using Deep Learning (CNN).

## 📁 Project Structure

```
lungvision/
├── app.py                  ← Flask backend
├── cancer.h5               ← Your trained model (place here)
├── requirements.txt
├── upload/                 ← Uploaded images (auto-created)
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── index.html          ← Home page
    ├── login.html          ← Login page
    ├── predict.html        ← CT scan upload & prediction
    ├── performance.html    ← Model performance analytics
    └── charts.html         ← Visualization dashboard
```

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your trained model
Copy your `cancer.h5` file into the `lungvision/` folder.

> **Note:** If `cancer.h5` is missing, the app runs in **demo mode** with random predictions — useful for UI testing.

### 3. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

## 🔐 Login Credentials
- **Username:** `admin`
- **Password:** `admin123`

To change credentials, set environment variables:
```bash
export APP_USER=youruser
export APP_PASS=yourpassword
export SECRET_KEY=your-random-secret-key
```

## 🧠 Model Details
- **Architecture:** CNN (4 Convolutional Blocks + Dense Head)
- **Input Size:** 150×150 RGB
- **Classes:** Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal
- **Framework:** TensorFlow / Keras

## 📊 Pages
| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page with features & stats |
| Login | `/login` | Authentication |
| Predict | `/predict` | Upload CT scan & get diagnosis |
| Performance | `/performance` | Model metrics & charts |
| Charts | `/charts` | Dataset & training visualizations |
| Logout | `/logout` | End session |

## ⚠️ Disclaimer
This system is for **educational purposes only**. All predictions must be reviewed by qualified medical professionals before any clinical decision-making.
