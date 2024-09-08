import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO

app = Flask(__name__)
model = load_model('model.h5')

# Define flower classes
classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 
           'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water-lily']

# Define flower recommendations
recommendations = {
    'astilbe': ['bellflower', 'carnation', 'coreopsis'],
    'bellflower': ['astilbe', 'calendula', 'rose'],
    'black_eyed_susan': ['california_poppy', 'sunflower', 'tulip'],
    'calendula': ['bellflower', 'carnation', 'dandelion'],
    'california_poppy': ['black_eyed_susan', 'coreopsis', 'iris'],
    'carnation': ['astilbe', 'calendula', 'rose'],
    'common_daisy': ['dandelion', 'sunflower', 'tulip'],
    'coreopsis': ['astilbe', 'california_poppy', 'water-lily'],
    'dandelion': ['common_daisy', 'calendula', 'tulip'],
    'iris': ['california_poppy', 'water-lily', 'rose'],
    'rose': ['bellflower', 'carnation', 'iris'],
    'sunflower': ['black_eyed_susan', 'common_daisy', 'tulip'],
    'tulip': ['black_eyed_susan', 'common_daisy', 'dandelion'],
    'water-lily': ['coreopsis', 'iris', 'rose']
}

# Endpoint to handle image prediction and recommendations
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Read the file into a BytesIO object
        img = load_img(BytesIO(file.read()), target_size=(64, 64))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]
        
        recommended_flowers = recommendations.get(predicted_class, [])
        
        return jsonify({'prediction': predicted_class, 'recommendations': recommended_flowers})

# Home endpoint to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
