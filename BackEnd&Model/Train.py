import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

class FaceRecognitionModel:
    def __init__(self, 
                 dataset_path, 
                 img_height=299, 
                 img_width=299, 
                 batch_size=32, 
                 num_classes=None,
                 dropout_rate=0.5,
                 l2_regularization=1e-4):
        """
        Initialize Face Recognition Model with Inception V3
        
        Parameters:
        - dataset_path: Path to the directory containing face images
        - img_height: Image height for resizing (default Inception V3 input)
        - img_width: Image width for resizing
        - batch_size: Training batch size
        - num_classes: Number of face classes/identities
        """
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        
        # Detect number of classes automatically
        if num_classes is None:
            self.num_classes = len(os.listdir(dataset_path))
        else:
            self.num_classes = num_classes
        
        # Model and training attributes
        self.model = None
        self.history = None
        self.best_model = None
        
    def _create_data_generators(self):
        """
        Create data generators with aggressive augmentation
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=(0.8, 1.2),
            fill_mode='nearest',
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def build_model(self, learning_rate=1e-4):
        """
        Build Inception V3 transfer learning model
        """
        base_model = InceptionV3(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze base model layers
        for layer in base_model.layers[-50:]:
            layer.trainable = False
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = Dense(
        # 512, 
        # activation='relu', 
        # kernel_regularizer=l2(self.l2_regularization)
        # )(x)
        # x = BatchNormalization()(x)
        # x = Dropout(self.dropout_rate)(x)
        
        # x = Dense(
        #     256, 
        #     activation='relu', 
        #     kernel_regularizer=l2(self.l2_regularization)
        # )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        predictions = Dense(
            self.num_classes, 
            activation='softmax',
        )(x)

        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()

        return self.model
    
    def train_with_k_fold(self, epochs=10, k_folds=5):
        """
        Train model using K-Fold Cross Validation
        """
        train_generator, validation_generator = self._create_data_generators()
        
        # K-Fold Cross Validation
        kf = KFold(n_splits=k_folds, shuffle=True)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            'best_face_recognition_model.keras',
            monitor='val_accuracy', 
            save_best_only=True
        )
        
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_generator.classes), 1):
            print(f"Training Fold {fold}")
            
            history = self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=[early_stopping, checkpoint]
            )
            
            fold_histories.append(history.history)
        
        # Load best model
        self.best_model = tf.keras.models.load_model('best_face_recognition_model.keras')
        
        return fold_histories
    
    def evaluate_model(self, validation_generator):
        """
        Evaluate model performance and generate reports
        """
        # Predictions
        predictions = self.best_model.predict(validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = validation_generator.classes
        
        # Classification Report
        class_report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=validation_generator.class_indices.keys()
        )
        print("Classification Report:\n", class_report)
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
    def plot_training_history(self, histories):
        """
        Plot training accuracy and loss
        """
        plt.figure(figsize=(12, 4))
        
        # Accuracy Plot
        plt.subplot(1, 2, 1)
        for history in histories:
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Loss Plot
        plt.subplot(1, 2, 2)
        for history in histories:
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

# Example Usage
def main():
    # Configuration
    dataset_path = "Celebrity Faces Dataset/"
    
    # Initialize and build model
    face_recognition = FaceRecognitionModel(dataset_path)
    model = face_recognition.build_model()
    
    # Train with K-Fold Cross Validation
    histories = face_recognition.train_with_k_fold()
    
    # Plot training history
    face_recognition.plot_training_history(histories)
    
    # Evaluate model
    train_generator, validation_generator = face_recognition._create_data_generators()
    face_recognition.evaluate_model(validation_generator)

if __name__ == "__main__":
    main()