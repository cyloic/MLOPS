#  Dog vs Cat Image Classifier

This project demonstrates a complete Machine Learning workflow using image data for a binary classification task — identifying whether an image is of a dog or a cat using traditional ML techniques (no deep learning). It includes model training, deployment via Flask and Streamlit, and performance testing using Locust.

The system classifies images of dogs and cats using handcrafted image features and traditional Scikit-learn classifiers. It covers all essential ML lifecycle components:

Feature extraction and model training

Model saving and loading

Real-time predictions through API and UI

Performance monitoring with Locust



- A **Flask API** (`api.py`) for serving predictions
- A **Streamlit UI** (`streamlit_app.py`) for interactive user input
- 

Demo Video: https://youtu.be/i0nwaQo4nFE

locust Report: https://docs.google.com/document/d/1fCrNrGn-qfE9UVK3dJK2qR98M1K-02eTcE7CqyFaEvI/edit?usp=sharing

---

## 📦 Project Structure

| Folder/File             | Description                             |
| ----------------------- | --------------------------------------- |
| `models/`               | Contains trained model and scaler files |
| ├── `dog_cat_model.pkl` | Serialized classifier model             |
| └── `scaler.pkl`        | Scikit-learn feature scaler             |
| `data/`                 | Training dataset                        |
| └── `train/`            | Labeled training images                 |
|     ├── `dogs/`         | Dog images                              |
|     └── `cats/`         | Cat images                              |
| `src/`                  | Source code for model and API           |
| ├── `api.py`            | Flask API server                        |
| ├── `model.py`          | Model training script (optional)        |
| ├── `preprocessing.py`  | Feature extraction utilities            |
| └── `prediction.py`     | Prediction helper function              |
| `streamlit_app.py`      | Streamlit web app                       |
| `README.md`             | This project guide                      |




---

## 🚀 Features


Handcrafted image features: color histograms, grayscale stats, and edge detection

Trained with Scikit-learn classifiers (e.g., Random Forest, Logistic Regression)

Interactive predictions via Streamlit UI

REST API using Flask for programmatic access

Load testing with Locust

---



## Run Locally

1. Clone the Repository
   
git clone https://github.com/cyloic/MLOPS.git
cd MLOPS

2. Install Dependencies
   
pip install -r requirements.txt

Typical dependencies include:

flask
streamlit
scikit-learn
opencv-python
numpy
pillow
joblib

##Model Information

The model was trained using:
Resized 32x32 RGB images

Handcrafted features including:
RGB histograms
Grayscale statistics
Sobel edge filters

The trained model and scaler are saved in the models/ directory.


##How to Run
Option 1: Run Flask API

cd src
python api.py
Then open: http://localhost:5000

Option 2: Run Streamlit App
streamlit run streamlit_app.py

## Upload an Image and get predictions

 Load Testing with Locust
Locust is used to simulate high traffic to test API performance.

Run Locust:
cd src
locust -f locustfile.py

Full Report: https://docs.google.com/document/d/1fCrNrGn-qfE9UVK3dJK2qR98M1K-02eTcE7CqyFaEvI/edit?usp=sharing


