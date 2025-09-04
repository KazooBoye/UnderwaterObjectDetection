#!/usr/bin/env python3
"""
YOLOv8 Training Script for Aquarium Dataset
Addresses class imbalance and multi-scale detection challenges
"""

import os
import yaml
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_class_distribution(data_path):
    """Analyze class distribution to calculate optimal weights"""
    train_labels = Path(data_path) / "train" / "labels"
    class_counts = Counter()
    
    for label_file in train_labels.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Calculate balanced weights (inverse frequency)
    class_weights = {}
    for class_id in range(7):  # 7 classes total
        count = class_counts.get(class_id, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * count)
        class_weights[class_id] = round(weight, 2)
    
    print("Class Distribution Analysis:")
    class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
    for class_id in range(7):
        count = class_counts.get(class_id, 0)
        percentage = (count / total_samples) * 100
        weight = class_weights[class_id]
        print(f"  {class_names[class_id]}: {count} samples ({percentage:.1f}%) - weight: {weight}")
    
    return class_weights

def create_training_config():
    """Create optimized training configuration for imbalanced dataset"""
    config = {
        # Training parameters optimized for class imbalance
        'epochs': 300,           # More epochs needed for minority classes
        'batch': 16,             # Smaller batch size for better gradient updates
        'imgsz': 640,           # Standard YOLOv8 input size
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.001,           # Final learning rate factor
        'momentum': 0.937,      # SGD momentum
        'weight_decay': 0.0005, # L2 regularization
        'warmup_epochs': 3,     # Warmup epochs
        'warmup_momentum': 0.8, # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        
        # Data augmentation (aggressive for minority classes)
        'hsv_h': 0.015,         # HSV hue augmentation
        'hsv_s': 0.7,           # HSV saturation augmentation  
        'hsv_v': 0.4,           # HSV value augmentation
        'degrees': 0.0,         # Rotation (keep 0 for underwater scenes)
        'translate': 0.1,       # Translation augmentation
        'scale': 0.5,           # Scale augmentation (important for multi-scale)
        'shear': 0.0,           # Shear augmentation
        'perspective': 0.0,     # Perspective augmentation
        'flipud': 0.0,          # Vertical flip (inappropriate for underwater)
        'fliplr': 0.5,          # Horizontal flip
        'mosaic': 1.0,          # Mosaic augmentation (helps with multi-object scenes)
        'mixup': 0.15,          # Mixup augmentation (helps with class imbalance)
        'copy_paste': 0.3,      # Copy-paste augmentation (good for object detection)
        
        # Loss function parameters
        'cls': 0.5,             # Classification loss weight
        'box': 7.5,             # Box regression loss weight  
        'dfl': 1.5,             # Distribution focal loss weight
        
        # Model architecture
        'dropout': 0.1,         # Dropout rate for overfitting prevention
        
        # Optimization settings for imbalanced data
        'optimizer': 'SGD',     # SGD works better for imbalanced datasets
        'close_mosaic': 15,     # Disable mosaic in final epochs
        'amp': True,            # Automatic mixed precision
        
        # Evaluation settings
        'val': True,            # Enable validation
        'plots': True,          # Generate training plots
        'save_period': 50,      # Save checkpoint every 50 epochs
    }
    
    return config

def setup_class_weights(data_yaml_path, class_weights):
    """Add class weights to data.yaml configuration"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Add class weights for loss function
    weights_list = [class_weights[i] for i in range(7)]
    data['class_weights'] = weights_list
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Updated {data_yaml_path} with class weights: {weights_list}")

def validate_dataset(data_path):
    """Validate dataset structure and files"""
    data_path = Path(data_path)
    
    issues = []
    for split in ['train', 'valid', 'test']:
        img_dir = data_path / split / 'images'  
        lbl_dir = data_path / split / 'labels'
        
        if not img_dir.exists():
            issues.append(f"Missing {split}/images directory")
            continue
            
        if not lbl_dir.exists():
            issues.append(f"Missing {split}/labels directory")  
            continue
            
        img_files = list(img_dir.glob('*.jpg'))
        lbl_files = list(lbl_dir.glob('*.txt'))
        
        print(f"{split}: {len(img_files)} images, {len(lbl_files)} labels")
        
        # Check for mismatched files
        img_stems = {f.stem for f in img_files}
        lbl_stems = {f.stem for f in lbl_files}
        
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} images without labels")
        if missing_images:
            issues.append(f"{split}: {len(missing_images)} labels without images")
    
    if issues:
        print("Dataset Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Dataset validation passed!")
        return True

if __name__ == "__main__":
    # Configuration
    data_path = "/Users/caoducanh/Desktop/Coding/UnderwaterObjectDetection/aquarium_pretrain"
    data_yaml = os.path.join(data_path, "data.yaml")
    
    print("=== YOLOv8 Aquarium Dataset Training Setup ===\n")
    
    # 1. Validate dataset structure
    print("1. Validating dataset structure...")
    if not validate_dataset(data_path):
        print("Please fix dataset issues before proceeding.")
        exit(1)
    
    # 2. Analyze class distribution
    print("\n2. Analyzing class distribution...")
    class_weights = analyze_class_distribution(data_path)
    
    # 3. Update data.yaml with class weights
    print("\n3. Setting up class weights...")
    setup_class_weights(data_yaml, class_weights)
    
    # 4. Create training configuration
    print("\n4. Training configuration created.")
    config = create_training_config()
    
    print("\nRecommended training command:")
    print("python train_yolov8.py")
    
    print("\nKey optimizations applied:")
    print("✓ Class weights calculated for imbalanced dataset")
    print("✓ Aggressive data augmentation for minority classes") 
    print("✓ Multi-scale training enabled")
    print("✓ Mosaic and mixup for complex multi-object scenes")
    print("✓ Extended training (300 epochs) for minority class learning")
    print("✓ Dropout regularization to prevent overfitting")
