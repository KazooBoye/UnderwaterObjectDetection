#!/usr/bin/env python3
"""
YOLOv8 Training Script for Aquarium Dataset
Optimized for class imbalance and multi-scale detection
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

def train_yolov8_aquarium():
    """Train YOLOv8 on aquarium dataset with optimized settings"""
    
    # Paths
    project_root = Path("/Users/caoducanh/Desktop/Coding/UnderwaterObjectDetection")
    data_yaml = project_root / "aquarium_pretrain" / "data.yaml"
    
    # Create runs directory
    runs_dir = project_root / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    # Initialize YOLOv8 model
    # Using YOLOv8n for faster training, can upgrade to YOLOv8s/m/l/x for better accuracy
    model = YOLO('yolov8n.pt')  # Load pretrained model
    
    print("=== YOLOv8 Training Configuration ===")
    print(f"Dataset: {data_yaml}")
    print(f"Model: YOLOv8n (nano)")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Output directory: {runs_dir}")
    
    # Training parameters optimized for class imbalance
    training_args = {
        # Basic training settings
        'data': str(data_yaml),
        'epochs': 300,           # Extended training for minority classes
        'batch': 16,             # Smaller batch size for stability
        'imgsz': 640,           # Standard input size
        'device': 0 if torch.cuda.is_available() else 'cpu',
        
        # Learning rate and optimization
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.001,           # Final learning rate factor  
        'momentum': 0.937,      # SGD momentum
        'weight_decay': 0.0005, # L2 regularization
        'warmup_epochs': 3,     # Warmup epochs
        'warmup_momentum': 0.8, # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        
        # Loss function weights (adjusted for multi-object scenes)
        'cls': 1.0,             # Classification loss weight
        'box': 7.5,             # Box regression loss weight
        'dfl': 1.5,             # Distribution focal loss weight
        
        # Data augmentation (aggressive for handling imbalance)
        'hsv_h': 0.015,         # HSV hue augmentation
        'hsv_s': 0.7,           # HSV saturation augmentation
        'hsv_v': 0.4,           # HSV value augmentation
        'degrees': 5.0,         # Small rotation for underwater scenes
        'translate': 0.1,       # Translation augmentation
        'scale': 0.5,           # Scale augmentation (crucial for multi-scale)
        'shear': 0.0,           # No shear for underwater objects
        'perspective': 0.0001,  # Minimal perspective for realism
        'flipud': 0.0,          # No vertical flip (inappropriate for fish)
        'fliplr': 0.5,          # Horizontal flip
        'mosaic': 1.0,          # Mosaic (excellent for multi-object scenes)
        'mixup': 0.15,          # Mixup (helps with class imbalance)
        'copy_paste': 0.3,      # Copy-paste augmentation
        
        # Training optimization
        'optimizer': 'SGD',     # SGD often better for imbalanced data
        'close_mosaic': 15,     # Disable mosaic in final epochs
        'amp': True,            # Automatic mixed precision
        'fraction': 1.0,        # Use full dataset
        'dropout': 0.1,         # Prevent overfitting
        
        # Validation and logging
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 50,      # Save every 50 epochs
        'cache': False,         # Don't cache (memory considerations)
        'workers': 4,           # Data loading workers
        'project': str(runs_dir),
        'name': 'aquarium_yolov8_balanced',
        'exist_ok': False,
        'pretrained': True,
        'verbose': True,
    }
    
    print("\n=== Starting Training ===")
    print("Key optimizations:")
    print("✓ Extended 300 epochs for minority class learning")
    print("✓ Multi-scale augmentation (scale=0.5)")
    print("✓ Mosaic + Mixup for complex scenes and class balance")  
    print("✓ Copy-paste augmentation for object detection")
    print("✓ Focal loss components (cls, box, dfl weights)")
    print("✓ Dropout regularization")
    print("✓ SGD optimizer for stable training on imbalanced data")
    
    # Start training
    try:
        results = model.train(**training_args)
        print(f"\n=== Training Completed Successfully ===")
        print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
        print(f"Last weights saved to: {results.save_dir}/weights/last.pt")
        return results
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None

def evaluate_model(model_path, data_yaml):
    """Evaluate trained model with focus on per-class metrics"""
    
    print(f"\n=== Evaluating Model ===")
    print(f"Model: {model_path}")
    
    # Load trained model
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data=str(data_yaml),
        split='test',  # Use test set for final evaluation
        imgsz=640,
        batch=16,
        conf=0.25,     # Confidence threshold
        iou=0.5,       # IoU threshold for NMS
        max_det=300,   # Maximum detections per image
        plots=True,    # Generate plots
    )
    
    print("\n=== Evaluation Results ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # Per-class results (crucial for imbalanced dataset)
    class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
    
    if hasattr(metrics.box, 'maps'):
        print("\nPer-class mAP50:")
        for i, (name, map_val) in enumerate(zip(class_names, metrics.box.maps)):
            print(f"  {name}: {map_val:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU (training will be slower)")
    
    # Run training
    results = train_yolov8_aquarium()
    
    if results:
        # Evaluate on test set
        project_root = Path("/Users/caoducanh/Desktop/Coding/UnderwaterObjectDetection")
        data_yaml = project_root / "aquarium_pretrain" / "data.yaml"
        best_model = results.save_dir / "weights" / "best.pt"
        
        if best_model.exists():
            evaluate_model(best_model, data_yaml)
        
        print(f"\n=== Training Summary ===")
        print(f"Training completed in {results.save_dir}")
        print(f"Best model: {results.save_dir}/weights/best.pt") 
        print(f"Training plots: {results.save_dir}")
        
        print(f"\n=== Next Steps ===")
        print("1. Check training plots for convergence")
        print("2. Analyze per-class performance (focus on minority classes)")
        print("3. Adjust confidence thresholds for balanced precision/recall")
        print("4. Consider ensemble methods if minority class performance is poor")
    else:
        print("Training failed. Please check error messages above.")
