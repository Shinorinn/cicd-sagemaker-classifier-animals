#!/usr/bin/env python3

import argparse
import json
import os
import tarfile
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def extract_model(model_dir):
    """Giải nén model.tar.gz nếu tồn tại."""
    tar_path = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=model_dir)
        print("Model extracted successfully.")

def find_model_path(base_dir):
    """Tìm đường dẫn chứa file saved_model.pb."""
    for root, dirs, files in os.walk(base_dir):
        if 'saved_model.pb' in files:
            return root
    return base_dir

def run_validation(model_dir, data_dir, output_dir, batch_size=32):
    """Hàm thực thi validation chính."""
    # 1. Chuẩn bị Model
    extract_model(model_dir)
    model_load_path = find_model_path(model_dir)
    print(f"Loading model from: {model_load_path}")
    model = tf.keras.models.load_model(model_load_path)
    
    # 2. Chuẩn bị Dữ liệu (Rescale về 0-1)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    # 3. Dự đoán
    print(f"Generating predictions for {val_generator.samples} samples...")
    predictions = model.predict(val_generator, verbose=1)
    
    # Logic xử lý Output
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    class_names = list(val_generator.class_indices.keys())

    # 4. Tính toán Metrics chuyên sâu
    acc = accuracy_score(true_classes, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, predicted_classes, average='weighted'
    )
    cm = confusion_matrix(true_classes, predicted_classes)
    report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)

    # 5. Lưu kết quả ra File
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu JSON tổng hợp
    results = {
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": report
    }
    
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- BÁO CÁO NHANH ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input/data')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    run_validation(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
def extract_model(model_dir):
    """Giải nén model.tar.gz nếu tồn tại."""
    tar_path = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=model_dir)
        print("Model extracted successfully.")

def find_model_path(base_dir):
    """Tìm đường dẫn chứa file saved_model.pb."""
    for root, dirs, files in os.walk(base_dir):
        if 'saved_model.pb' in files:
            return root
    return base_dir

def run_validation(model_dir, data_dir, output_dir, batch_size=32):
    """Hàm thực thi validation chính."""
    # 1. Chuẩn bị Model
    extract_model(model_dir)
    model_load_path = find_model_path(model_dir)
    print(f"Loading model from: {model_load_path}")
    model = tf.keras.models.load_model(model_load_path)
    
    # 2. Chuẩn bị Dữ liệu (Giữ nguyên 0-255 vì model đã có layer Rescale)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary', # Chó/Mèo dùng binary là chuẩn nhất
        shuffle=False
    )
    
    # 3. Dự đoán
    print(f"Generating predictions for {val_generator.samples} samples...")
    predictions = model.predict(val_generator, verbose=1)
    
    # Logic xử lý Output: Quan trọng nhất để tránh lỗi Accuracy 0.5
    if predictions.shape[1] == 1:
        # Trường hợp 1 node đầu ra (Sigmoid)
        predicted_classes = (predictions.flatten() > 0.5).astype(int)
    else:
        # Trường hợp 2 node đầu ra (Softmax)
        predicted_classes = np.argmax(predictions, axis=1)
        
    true_classes = val_generator.classes
    class_names = list(val_generator.class_indices.keys())

    # 4. Tính toán Metrics chuyên sâu
    acc = accuracy_score(true_classes, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, predicted_classes, average='weighted'
    )
    cm = confusion_matrix(true_classes, predicted_classes)
    report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)

    # 5. Lưu kết quả ra File
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu JSON tổng hợp
    results = {
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": report
    }
    
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- BÁO CÁO NHANH ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Results saved to: {output_dir}")
    print("File: evaluation.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input/data')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    run_validation(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )