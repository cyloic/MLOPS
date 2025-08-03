import numpy as np
import keras
from PIL import Image
import io
import base64
from pathlib import Path
import time
import json
from datetime import datetime
import cv2

class DogCatPredictor:
    """
    Handles predictions for the dog vs cat classifier
    """
    
    def __init__(self, model_path="models/dog_cat_cnn_latest.h5", img_size=(224, 224)):
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.model = None
        self.class_names = ['Cat', 'Dog']
        
        # Load model if it exists
        self.load_model()
    
    def load_model(self, model_path=None):
        """Load the trained model"""
        if model_path:
            self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            print(f"Model file not found: {self.model_path}")
            print("Please train a model first using model.py")
            return False
        
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for prediction
        Args:
            image_input: Can be file path, PIL Image, or numpy array
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = Image.fromarray((image_input * 255).astype(np.uint8))
            elif hasattr(image_input, 'read'):
                # File-like object (e.g., uploaded file)
                image = Image.open(image_input)
            else:
                # Assume PIL Image
                image = image_input
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.img_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict_single(self, image_input, return_probabilities=False):
        """
        Predict a single image
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image_input)
        if processed_image is None:
            return {"error": "Failed to preprocess image"}
        
        try:
            # Make prediction
            prediction_prob = self.model.predict(processed_image, verbose=0)[0][0]
            
            # Convert to class prediction
            predicted_class_idx = int(prediction_prob > 0.5)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            result = {
                'prediction': predicted_class,
                'confidence': confidence,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'Cat': float(1 - prediction_prob),
                    'Dog': float(prediction_prob)
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, image_list):
        """
        Predict multiple images at once
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        results = []
        start_time = time.time()
        
        for i, image_input in enumerate(image_list):
            result = self.predict_single(image_input, return_probabilities=True)
            result['image_index'] = i
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'predictions': results,
            'batch_size': len(image_list),
            'total_processing_time_ms': round(total_time, 2),
            'avg_processing_time_ms': round(total_time / len(image_list), 2)
        }
    
    def predict_from_base64(self, base64_string):
        """
        Predict from base64 encoded image
        """
        try:
            # Decode base64 string
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            return self.predict_single(image, return_probabilities=True)
            
        except Exception as e:
            return {"error": f"Failed to process base64 image: {str(e)}"}
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            'model_path': str(self.model_path),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'class_names': self.class_names
        }

# API Helper Functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_file(file):
    """Process uploaded file for prediction"""
    if file and allowed_file(file.filename):
        try:
            # Read image
            image = Image.open(file.stream)
            return image
        except Exception as e:
            return None
    return None

# Batch processing functions
class BatchProcessor:
    """
    Handle batch processing of images for retraining
    """
    
    def __init__(self, upload_dir="data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.cats_dir = self.upload_dir / "cats"
        self.dogs_dir = self.upload_dir / "dogs"
        self.unlabeled_dir = self.upload_dir / "unlabeled"
        
        self.cats_dir.mkdir(exist_ok=True)
        self.dogs_dir.mkdir(exist_ok=True)
        self.unlabeled_dir.mkdir(exist_ok=True)
    
    def save_uploaded_files(self, files, label=None):
        """
        Save uploaded files to appropriate directories
        Args:
            files: List of uploaded files
            label: 'cat', 'dog', or None for unlabeled
        """
        saved_files = []
        errors = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{file.filename}"
                    
                    # Determine save directory
                    if label == 'cat':
                        save_path = self.cats_dir / filename
                    elif label == 'dog':
                        save_path = self.dogs_dir / filename
                    else:
                        save_path = self.unlabeled_dir / filename
                    
                    # Save file
                    file.save(str(save_path))
                    saved_files.append(str(save_path))
                    
                except Exception as e:
                    errors.append(f"Failed to save {file.filename}: {str(e)}")
            else:
                errors.append(f"Invalid file: {file.filename}")
        
        return {
            'saved_files': saved_files,
            'errors': errors,
            'total_saved': len(saved_files)
        }
    
    def get_uploaded_stats(self):
        """Get statistics about uploaded files"""
        cats_count = len(list(self.cats_dir.glob("*")))
        dogs_count = len(list(self.dogs_dir.glob("*")))
        unlabeled_count = len(list(self.unlabeled_dir.glob("*")))
        
        return {
            'cats': cats_count,
            'dogs': dogs_count,
            'unlabeled': unlabeled_count,
            'total': cats_count + dogs_count + unlabeled_count
        }

# Performance monitoring
class PerformanceMonitor:
    """
    Monitor model performance and system metrics
    """
    
    def __init__(self):
        self.prediction_times = []
        self.prediction_confidences = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = time.time()
    
    def log_prediction(self, processing_time, confidence, success=True):
        """Log a prediction result"""
        if success:
            self.prediction_times.append(processing_time)
            self.prediction_confidences.append(confidence)
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.prediction_times:
            return {
                'total_predictions': 0,
                'success_rate': 0,
                'avg_processing_time': 0,
                'avg_confidence': 0
            }
        
        return {
            'total_predictions': self.success_count + self.error_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / (self.success_count + self.error_count),
            'avg_processing_time_ms': np.mean(self.prediction_times),
            'min_processing_time_ms': np.min(self.prediction_times),
            'max_processing_time_ms': np.max(self.prediction_times),
            'avg_confidence': np.mean(self.prediction_confidences),
            'uptime_hours': (time.time() - self.start_time) / 3600
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DogCatPredictor()
    
    # Test prediction with a sample image
    # Make sure you have a test image in your project
    test_image_path = "data/test/sample_image.jpg"
    
    if Path(test_image_path).exists():
        result = predictor.predict_single(test_image_path, return_probabilities=True)
        print("Prediction result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {test_image_path}")
        print("Model is ready for predictions!")
    
    # Get model info
    model_info = predictor.get_model_info()
    print("\nModel Information:")
    print(json.dumps(model_info, indent=2))