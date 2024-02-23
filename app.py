# app.py
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

app = Flask(__name__)

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the image file from the request
    file = request.files['image']
    img = image.load_img(file, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Prepare predictions to display
    results = []
    for _, label, confidence in decoded_predictions:
        results.append({'label': label, 'confidence': f"{confidence * 100:.2f}%"})

    return {'predictions': results}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
