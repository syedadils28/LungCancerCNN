from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil

app = Flask(__name__)
app.secret_key = 'lungcancer123'

# Load trained model
model = load_model('cancer.h5')

# Class labels (must match training order from class_indices)
CLASS_NAMES = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Dummy login credentials
USERNAME = 'admin'
PASSWORD = 'admin123'

# Ensure upload folder exists
os.makedirs('upload', exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('preview'))
        else:
            return render_template('login.html', error='Invalid credentials! Try admin / admin123')
    return render_template('login.html', error=None)


@app.route('/preview', methods=['GET', 'POST'])
def preview():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    prediction = None
    img_filename = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            # Save uploaded image
            filepath = os.path.join('upload', file.filename)
            file.save(filepath)

            # Also copy to static so it can be served
            static_copy = os.path.join('static', 'uploaded_preview.png')
            shutil.copy(filepath, static_copy)
            img_filename = 'uploaded_preview.png'

            # Preprocess image
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            result = model.predict(img_array)
            class_idx = int(np.argmax(result))
            predicted_class = CLASS_NAMES[class_idx]
            confidence = float(np.max(result)) * 100

            if predicted_class == 'Normal':
                prediction = {
                    'label': 'Normal (Non-Cancerous)',
                    'type': 'normal',
                    'confidence': f'{confidence:.1f}%',
                    'detail': 'No signs of lung cancer detected.'
                }
            else:
                prediction = {
                    'label': f'Cancerous — {predicted_class}',
                    'type': 'cancer',
                    'confidence': f'{confidence:.1f}%',
                    'detail': f'Type identified: {predicted_class}'
                }

            return render_template('preview.html', prediction=prediction, img_filename=img_filename)

    return render_template('preview.html', prediction=None, img_filename=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
