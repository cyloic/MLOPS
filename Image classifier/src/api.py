from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import joblib
import cv2

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('../models/dog_cat_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    model = None
    scaler = None
    print(f"❌ Could not load model: {e}")

def extract_features(image):
    """Simple flattening to match model expectations (4096 features)"""
    img_resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    return gray.flatten()

# HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dog vs Cat Classifier</title>
    <style>
        body { font-family: Arial; margin: 40px; text-align: center; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .upload-box { border: 2px dashed #007bff; padding: 40px; margin: 20px 0; border-radius: 10px; background: #f8f9fa; }
        button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin: 20px 0; padding: 20px; background: #e8f5e8; border-radius: 5px; }
        .tech-info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐱🐶 Dog vs Cat Classifier</h1>
        <div class="tech-info">
            <strong>Traditional ML model</strong><br>
            Based on raw grayscale pixel values (flattened to 4096 features)
        </div>
        <div class="upload-box">
            <form action="/predict" method="post" enctype="multipart/form-data">
                <p>📁 Choose an image of a dog or cat</p>
                <input type="file" name="file" accept="image/*" required style="margin: 10px;">
                <br><br>
                <button type="submit">🔍 Predict!</button>
            </form>
        </div>
        <p><em>Upload a JPG image to get a prediction!</em></p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        file = request.files['file']
        img = Image.open(file)
        img = img.convert('RGB')
        img_array = np.array(img)

        # Extract features
        features = extract_features(img_array)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0]

        if prediction == 1:
            result = f"🐶 DOG (Confidence: {confidence[1]:.2%})"
        else:
            result = f"🐱 CAT (Confidence: {confidence[0]:.2%})"

        return f"""
        <html>
        <body style="font-family: Arial; text-align: center; margin: 40px; background: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h1>🎯 Prediction Result</h1>
                <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h2>{result}</h2>
                </div>
                <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; font-size: 14px;">
                    <strong>Model:</strong> Scikit-learn with 4096 grayscale features
                </div>
                <a href="/" style="text-decoration: none;">
                    <button style="background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 16px;">
                        🔄 Try Another Image
                    </button>
                </a>
            </div>
        </body>
        </html>
        """

    except Exception as e:
        return f"""
        <html>
        <body style="font-family: Arial; text-align: center; margin: 40px;">
            <h1>❌ Error</h1>
            <p>Could not process image: {str(e)}</p>
            <a href="/" style="text-decoration: none;">
                <button style="background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px;">
                    Try Again
                </button>
            </a>
        </body>
        </html>
        """

if __name__ == '__main__':
    app.run(debug=True, port=5000)
