# 🫁 Lung Cancer Detection — CNN + Flask

A web app that classifies chest CT scan images into:
- **Adenocarcinoma**
- **Large Cell Carcinoma**
- **Squamous Cell Carcinoma**
- **Normal**

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Get it from: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

Extract into `model/` so you have `model/train/`, `model/valid/`, `model/test/`

### 3. Train the model
```bash
cd model
python train_model.py
```
This saves `cancer.h5` in the root folder. Takes ~20–40 minutes.

### 4. Run the app
```bash
python app.py
```
Open: http://127.0.0.1:5000

**Login:** `admin` / `admin123`

---

## Project Structure
```
LungCancerCNN/
├── model/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── train_model.py   ← run this to train
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   ├── login.html
│   └── preview.html
├── upload/              ← uploaded images saved here
├── app.py               ← Flask app
├── cancer.h5            ← trained model (generated after training)
├── requirements.txt
└── README.md
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `cancer.h5 not found` | Run `train_model.py` first |
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow==2.10.0` |
| Port 5000 in use | Change to `app.run(port=5001)` in app.py |
| Low accuracy | Increase epochs to 30 in `train_model.py` |

> ⚕️ For educational purposes only. Not a substitute for medical diagnosis.
