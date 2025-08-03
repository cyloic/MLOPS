import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm

class ImagePreprocessor:
    """
    Handles all data preprocessing for dog vs cat classification
    """
    
    def __init__(self, data_dir="data", img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
        # Create directories if they don't exist
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.images = []
        self.labels = []
        self.image_paths = []
        
    def load_and_preprocess_images(self, folder_path):
        """
        Load images from folder and preprocess them
        Expected structure: folder_path/cats/*.jpg, folder_path/dogs/*.jpg
        """
        print("Loading and preprocessing images...")
        
        # Check if cats and dogs folders exist
        cats_folder = Path(folder_path) / "cats"
        dogs_folder = Path(folder_path) / "dogs"
        
        if not cats_folder.exists() or not dogs_folder.exists():
            print(f"Creating folder structure in {folder_path}")
            cats_folder.mkdir(exist_ok=True)
            dogs_folder.mkdir(exist_ok=True)
            print("Please put cat images in 'cats' folder and dog images in 'dogs' folder")
            return
        
        # Load cat images
        cat_images = list(cats_folder.glob("*.jpg")) + list(cats_folder.glob("*.jpeg")) + list(cats_folder.glob("*.png"))
        for img_path in tqdm(cat_images, desc="Loading cat images"):
            if self._load_single_image(img_path, label=0):  # 0 for cat
                self.image_paths.append(str(img_path))
        
        # Load dog images
        dog_images = list(dogs_folder.glob("*.jpg")) + list(dogs_folder.glob("*.jpeg")) + list(dogs_folder.glob("*.png"))
        for img_path in tqdm(dog_images, desc="Loading dog images"):
            if self._load_single_image(img_path, label=1):  # 1 for dog
                self.image_paths.append(str(img_path))
        
        print(f"Loaded {len(self.images)} images total")
        print(f"Cats: {sum(1 for label in self.labels if label == 0)}")
        print(f"Dogs: {sum(1 for label in self.labels if label == 1)}")
        
        return np.array(self.images), np.array(self.labels)
    
    def _load_single_image(self, img_path, label):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = Image.open(img_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.img_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0
            
            self.images.append(img_array)
            self.labels.append(label)
            
            return True
            
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return False
    
    def create_data_generators(self, validation_split=0.2, batch_size=32):
        """
        Create data generators for training with augmentation
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(validation_split=validation_split)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if len(self.images) == 0:
            print("No images loaded. Please run load_and_preprocess_images first.")
            return None, None, None, None
        
        X = np.array(self.images)
        y = np.array(self.labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        return X_train, X_test, y_train, y_test
    
    def visualize_data_distribution(self):
        """Visualize the distribution of cats vs dogs"""
        if len(self.labels) == 0:
            print("No data loaded yet.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Count distribution
        labels_count = pd.Series(self.labels).value_counts()
        labels_names = ['Cats', 'Dogs']
        
        plt.subplot(1, 2, 1)
        plt.bar(labels_names, labels_count.values, color=['orange', 'skyblue'])
        plt.title('Distribution of Images')
        plt.ylabel('Number of Images')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(labels_count.values, labels=labels_names, autopct='%1.1f%%', 
                colors=['orange', 'skyblue'])
        plt.title('Proportion of Cats vs Dogs')
        
        plt.tight_layout()
        plt.show()
    
    def show_sample_images(self, num_samples=8):
        """Display sample images from the dataset"""
        if len(self.images) == 0:
            print("No images loaded yet.")
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        # Get random samples
        indices = np.random.choice(len(self.images), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            axes[i].imshow(self.images[idx])
            label = "Dog" if self.labels[idx] == 1 else "Cat"
            axes[i].set_title(f'{label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_image_statistics(self):
        """Get statistics about the images"""
        if len(self.images) == 0:
            print("No images loaded yet.")
            return
        
        images_array = np.array(self.images)
        
        stats = {
            'total_images': len(self.images),
            'image_shape': images_array.shape[1:],
            'cats_count': sum(1 for label in self.labels if label == 0),
            'dogs_count': sum(1 for label in self.labels if label == 1),
            'mean_pixel_value': np.mean(images_array),
            'std_pixel_value': np.std(images_array)
        }
        
        print("Dataset Statistics:")
        print(f"Total images: {stats['total_images']}")
        print(f"Image shape: {stats['image_shape']}")
        print(f"Cats: {stats['cats_count']}")
        print(f"Dogs: {stats['dogs_count']}")
        print(f"Mean pixel value: {stats['mean_pixel_value']:.3f}")
        print(f"Std pixel value: {stats['std_pixel_value']:.3f}")
        
        return stats
    
    def preprocess_new_image(self, image_path):
        """Preprocess a single new image for prediction"""
        try:
            image = Image.open(image_path)
            
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
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None

# Helper function to organize your data
def organize_data_folders(source_folder, train_folder):
    """
    Helper function to organize your images into proper folder structure
    Call this if your images are not organized in cats/dogs folders
    """
    source_path = Path(source_folder)
    train_path = Path(train_folder)
    
    # Create train folder structure
    cats_folder = train_path / "cats"
    dogs_folder = train_path / "dogs"
    cats_folder.mkdir(parents=True, exist_ok=True)
    dogs_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Created folder structure at {train_path}")
    print("Please manually organize your images:")
    print(f"- Put cat images in: {cats_folder}")
    print(f"- Put dog images in: {dogs_folder}")
    
    return str(cats_folder), str(dogs_folder)

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # If you need to organize your data first
    # organize_data_folders("path/to/your/images", "data/train")
    
    # Load and preprocess images
    X, y = preprocessor.load_and_preprocess_images("data/train")
    
    if X is not None:
        # Show statistics
        preprocessor.get_image_statistics()
        
        # Visualize data
        preprocessor.visualize_data_distribution()
        preprocessor.show_sample_images()
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data()
        
        print("Data preprocessing completed!")