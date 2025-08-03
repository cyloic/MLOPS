import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.applications import VGG16, ResNet50
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime

class DogCatCNN:
    """
    CNN Model for Dog vs Cat Classification
    """
    
    def __init__(self, input_shape=(224, 224, 3), model_name="dog_cat_cnn"):
        self.input_shape = input_shape
        self.model_name = model_name
        self.model = None
        self.history = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
    def create_model(self):
        """Create CNN architecture"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Flatten and Dense layers
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model created successfully!")
        self.model.summary()
        
        return model
    
    def train(self, train_generator, validation_generator, epochs=50):
        """Train the model"""
        if self.model is None:
            print("Model not created. Creating model first...")
            self.create_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_dir / f"{self.model_name}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training...")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        # Save final model
        self.save_model()
        
        return self.history
    
    def train_with_arrays(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train with numpy arrays instead of generators"""
        if self.model is None:
            print("Model not created. Creating model first...")
            self.create_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_dir / f"{self.model_name}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training with arrays...")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        # Save final model
        self.save_model()
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and return metrics"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None
        
        print("Evaluating model...")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("="*50)
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
        return metrics, y_pred, y_pred_prob
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Cat', 'Dog'],
                   yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_image(self, image_array):
        """Predict a single image"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None
        
        # Make prediction
        prediction_prob = self.model.predict(image_array)[0][0]
        prediction = "Dog" if prediction_prob > 0.5 else "Cat"
        confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probability': float(prediction_prob)
        }
    
    def predict_batch(self, images):
        """Predict multiple images"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None
        
        predictions = self.model.predict(images)
        results = []
        
        for prob in predictions:
            pred_prob = prob[0]
            prediction = "Dog" if pred_prob > 0.5 else "Cat"
            confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
            
            results.append({
                'prediction': prediction,
                'confidence': float(confidence),
                'probability': float(pred_prob)
            })
        
        return results
    
    def save_model(self, custom_path=None):
        """Save the trained model"""
        if self.model is None:
            print("No model to save.")
            return
        
        if custom_path:
            model_path = Path(custom_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"{self.model_name}_{timestamp}.h5"
        
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Also save as the latest model
        latest_path = self.model_dir / f"{self.model_name}_latest.h5"
        self.model.save(latest_path)
        print(f"Latest model saved to: {latest_path}")
        
        return str(model_path)
    
    def load_model(self, model_path=None):
        """Load a trained model"""
        if model_path is None:
            # Try to load the latest model
            model_path = self.model_dir / f"{self.model_name}_latest.h5"
        
        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            self.model = load_model(model_path)
            print(f"Model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "No model loaded"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

# Transfer Learning Model (Alternative approach)
class TransferLearningModel(DogCatCNN):
    """
    Transfer learning model using pre-trained networks
    """
    
    def __init__(self, base_model='VGG16', input_shape=(224, 224, 3), model_name="transfer_dog_cat"):
        super().__init__(input_shape, model_name)
        self.base_model_name = base_model
    
    def create_model(self):
        """Create transfer learning model"""
        # Load pre-trained model
        if self.base_model_name == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        model = Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"Transfer learning model created with {self.base_model_name}!")
        self.model.summary()
        
        return model

# Example usage
if __name__ == "__main__":
    # Create model instance
    cnn_model = DogCatCNN()
    
    # Create the model architecture
    model = cnn_model.create_model()
    
    print("Model created! Ready for training.")
    print("Use the preprocessing.py to prepare your data first.")