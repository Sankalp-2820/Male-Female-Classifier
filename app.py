import os
# import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from tensorflow.keras.models import load_model
import cv2
# from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/sankalpdeshmukh2820/Desktop/ML/Male Female Classifier/Male-Female-Classifier/Uploads_folder'
# scaler = pickle.load(open('scaling.pkl','rb'))
# model  = pickle.load(open('CNNmodel.pkl','rb'))
model = load_model('myModel.h5')


def predict_gender(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    # img = Image.open(image_path)
    # img = img.resize((64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    # print(f"{result[0][0]*100}%")
    if result >= 0.5:
        return 'Male Detected',result[0][0]
    else:
        return 'Female Detected',result[0][0]


@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)
            gender,final = predict_gender(image_path)
            return render_template('index.html', gender=gender, image_path=image_path,final = final)
    return render_template('index.html', gender=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
