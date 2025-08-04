import streamlit as st
import requests
from PIL import Image

st.title("Cat vs Dog Classifier with Retrain")

# Prediction section
uploaded_file = st.file_uploader("Upload an image of a Cat or Dog", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Sending image to API for prediction..."):
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": uploaded_file.getvalue()}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction']}**")
            else:
                st.error(f"Error from server: {response.text}")

st.markdown("---")

# Retrain section
st.header("Retrain the model with new images")

uploaded_files = st.file_uploader(
    "Upload images for retraining", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)
label = st.radio("Select label for uploaded images:", options=["Cat", "Dog"])

if st.button("Retrain Model"):
    if not uploaded_files:
        st.error("Please upload at least one image for retraining.")
    else:
        with st.spinner("Retraining model, please wait..."):
            # Prepare files for upload
            files = []
            for file in uploaded_files:
                files.append(
                    ("files", (file.name, file, file.type))
                )
            # label as form data
            data = {"label": label}

            retrain_response = requests.post(
                "http://localhost:8000/retrain",
                data=data,
                files=files
            )
            if retrain_response.status_code == 200:
                st.success("Model retrained successfully!")
            else:
                st.error(f"Retrain failed: {retrain_response.text}")
