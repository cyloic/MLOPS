import streamlit as st
import numpy as np
from PIL import Image
import joblib
import cv2

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../models/dog_cat_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"❌ Could not load model or scaler: {e}")
        return None, None

model, scaler = load_model()

# Feature extraction function
def extract_features(image):
    img_resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    return gray.flatten()

# App UI
st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐶🐱")
st.title("🐶🐱 Dog vs Cat Classifier (Scikit-learn)")
st.markdown("""
This app uses a **traditional ML model** trained on flattened grayscale image pixels.<br>
Upload an image of a dog or a cat to see the prediction.
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert and extract features
    img_np = np.array(image)
    features = extract_features(img_np)

    if model is not None and scaler is not None:
        try:
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            confidence = model.predict_proba(features_scaled)[0]

            if prediction == 1:
                st.success(f"🐶 It's a **DOG** with {confidence[1]:.2%} confidence!")
            else:
                st.success(f"🐱 It's a **CAT** with {confidence[0]:.2%} confidence!")

            st.info(f"Extracted {features.shape[0]} features from the image.")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning("Model not loaded correctly.")
