import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Eksperimen
class FaceRecognitionConfig:
    TRAIN_DIR = 'Celebrity Faces Dataset/'
    IMG_HEIGHT = 299
    IMG_WIDTH = 299
    BATCH_SIZE = 32  # Naikkan batch size
    EPOCHS = 50  # Perpanjang epoch
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0
    NUM_CLASSES = len(os.listdir(TRAIN_DIR))

# Advanced Data Augmentation
def create_advanced_data_generator():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        validation_split=0.2  # 20% for validation
    )
    
    return train_datagen

# Improved Model Architecture
def create_enhanced_inception_model(config):
    # Load base model
    base_model = InceptionV3(
        weights='imagenet', 
        include_top=False, 
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    
     # Freeze base model layers
    for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    output = Dense(FaceRecognitionConfig.NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)

    model.summary()
    
    return model

# Advanced Callbacks
def create_advanced_callbacks(config):
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=8, 
        min_lr=1e-6,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model_train.keras',
        monitor='val_accuracy', 
        save_best_only=True,
        verbose=1
    )
    
    csv_logger = CSVLogger('train.csv')
    
    return [reduce_lr, early_stopping, model_checkpoint, csv_logger]

# Compile dan Training Model
def train_face_recognition_model():
    # Konfigurasi
    config = FaceRecognitionConfig()
    
    # Data Generator
    data_generator = create_advanced_data_generator()
    
    # Prepare Generator
    train_generator = data_generator.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = data_generator.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Model
    model = create_enhanced_inception_model(config)
    
    # Optimizers
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    
    # Compile Model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Callbacks
    callbacks = create_advanced_callbacks(config)
    
    # Training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )
    
    # Evaluasi Model
    def plot_training_history(history):
        plt.figure(figsize=(15, 5))
        
        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_performance.png')
    
    # Confusion Matrix
    def plot_confusion_matrix(model, validation_generator):
        # Prediksi
        validation_generator.reset()
        predictions = model.predict(validation_generator, steps=len(validation_generator))
        y_pred = np.argmax(predictions, axis=1)
        y_true = validation_generator.classes
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
    
    # Generate Visualisasi
    plot_training_history(history)
    plot_confusion_matrix(model, validation_generator)
    
    # Classification Report
    y_pred = np.argmax(model.predict(validation_generator), axis=1)
    class_labels = list(validation_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(validation_generator.classes, y_pred, target_names=class_labels))
    
    return model, history

# Jalankan Training
model, history = train_face_recognition_model()