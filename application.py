import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('my_model.h5')  # Load your trained model

def predict_gender(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    if result > 0.5:
        return "Male"
    else:
        return "Female"

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)
            gender = predict_gender(image_path)
            return render_template('index.html', gender=gender, image_path=image_path)
    return render_template('index.html', gender=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
