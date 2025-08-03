#  Dog vs Cat Image Classifier

This project is a simple image classification app that predicts whether an uploaded image is of a **dog** or a **cat**, using traditional machine learning techniques (not deep learning). It includes both:

- A **Flask API** (`api.py`) for serving predictions
- A **Streamlit UI** (`streamlit_app.py`) for interactive user input
- 

Demo Video: https://youtu.be/GpGCMKrQE6E

locust Report: https://docs.google.com/document/d/1fCrNrGn-qfE9UVK3dJK2qR98M1K-02eTcE7CqyFaEvI/edit?usp=sharing

---

## 📦 Project Structure

Image-classifier/
│
├── models/
│ ├── dog_cat_model.pkl # Trained Scikit-learn model
│ └── scaler.pkl # Feature scaler (StandardScaler)
│
├── data/
│ └── train/
│ ├── cats/ # Training images - cats
│ └── dogs/ # Training images - dogs
│
├── src/
│ ├── api.py # Flask API server
│ ├── preprocessing.py # Feature extraction logic
│ ├── model.py # Training script (optional)
│ └── prediction.py # CLI/Script for testing predictions
│
├── streamlit_app.py # Streamlit UI interface
└── README.md



---

## 🚀 Features

- Uses **Scikit-learn** models with hand-crafted features (no CNN)
- Extracts color, grayscale, statistical, and edge-based features
- Predicts label and confidence score for input image
- Supports both **Flask API** and **Streamlit UI**

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt

## Typical dependencies include:

flask
streamlit
scikit-learn
numpy
opencv-python
pillow
joblib


 ## Model Info
The model was trained using:

Resized 64x64 RGB images

Histograms, grayscale stats, Sobel edges

Traditional classifiers (e.g., RandomForest, Logistic Regression)

Model and scaler are saved in the models/ directory.


## Run Locally

cd src
python api.py
Open your browser at: http://localhost:5000

Option 2: Streamlit App
Run the Streamlit app:
streamlit run streamlit_app.py


## Upload an Image and get predictions
