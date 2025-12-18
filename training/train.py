import os
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

save_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

def load_and_train(train_dir, save_dir, epochs, batch_size, lr):
    """
    Load data, train model, và evaluate
    """
    # Data preprocessing
    train_datagen = ImageDataGenerator()
    
    # test_datagen = ImageDataGenerator()
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    
    # Load test data
    # validation_generator = test_datagen.flow_from_directory(
    #     test_dir,
    #     target_size=(224, 224),
    #     batch_size=32,
    #     class_mode='binary'
    # )
    
    # Build model with ResNet50V2 pre-trained weights
    base_model = keras.applications.ResNet50V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom top layers with rescaling
    model = keras.Sequential([
        layers.Rescaling(1./255.),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        train_generator,
        epochs=epochs,
        # validation_data=validation_generator
    )
    
    # Evaluate
    # val_loss, val_accuracy = model.evaluate(validation_generator)
    
    # print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model as SavedModel format (better for production)
    model_save_path = os.path.join(save_dir, '001')
    model.save(model_save_path, save_format='tf')
    
    # Save metrics
    # metrics = {
    #     'val_accuracy': float(val_accuracy),
    #     'val_loss': float(val_loss),
    #     'train_accuracy': float(history.history['accuracy'][-1])
    # }
    
    # with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
    #     json.dump(metrics, f)
    
    # return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='/opt/ml/input/data/train')
    # parser.add_argument('--test-dir', type=str, default='/opt/ml/input/data/test')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    metrics = load_and_train(args.train_dir, save_dir, args.epochs, args.batch_size, args.learning_rate)
    print("Training completed!")