from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import joblib
from typing import List
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your existing model and scaler
model = joblib.load("../models/dog_cat_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def extract_features(image):
    # Resize to 32x32 as you said earlier, convert to grayscale, flatten
    img_resized = cv2.resize(image, (32, 32))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    return gray.flatten()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        features = extract_features(image)
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        return {"prediction": "Cat" if pred == 0 else "Dog"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/retrain")
async def retrain(
    label: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        save_dir = f"retrain_data/{label.lower()}"
        os.makedirs(save_dir, exist_ok=True)

        # Save uploaded files locally
        for file in files:
            file_path = os.path.join(save_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # TODO: Add your retraining logic here using images saved at save_dir

        return {"message": f"Received {len(files)} files for retraining as {label}"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
