import os
import random
from locust import HttpUser, task, between

class ImagePredictionUser(HttpUser):
    wait_time = between(1, 3)  # wait between 1 to 3 seconds between tasks

    def on_start(self):
        # Folder where test images are stored
        self.image_folder = "sample_images"
        try:
            self.image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not self.image_files:
                print(f"⚠️ No images found in folder '{self.image_folder}' for load testing.")
        except FileNotFoundError:
            print(f"❌ Folder '{self.image_folder}' does not exist. Please create it and add images.")
            self.image_files = []

    @task
    def predict_image(self):
        if not self.image_files:
            return  # Skip task if no images

        image_name = random.choice(self.image_files)
        image_path = os.path.join(self.image_folder, image_name)

        with open(image_path, "rb") as img_file:
            files = {"file": (image_name, img_file, "image/jpeg")}
            self.client.post("/predict", files=files)
