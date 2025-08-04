import os
import random
from locust import HttpUser, task, between

class CatDogUser(HttpUser):
    host = "http://localhost:8000"  # <-- Add this!
    wait_time = between(1, 3)

    @task
    def predict_image(self):
        image_folder = os.path.join(os.path.dirname(__file__), "sample_images")

        if not os.path.exists(image_folder):
            print(f"❌ Folder does not exist: {image_folder}")
            return

        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print(f"⚠️ No image files found in {image_folder}")
            return

        selected_image = random.choice(image_files)
        image_path = os.path.join(image_folder, selected_image)

        with open(image_path, "rb") as img_file:
            files = {"file": (selected_image, img_file, "image/jpeg")}
            response = self.client.post("/predict", files=files)

            if response.status_code != 200:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            else:
                print(f"✅ Prediction success for {selected_image}")
