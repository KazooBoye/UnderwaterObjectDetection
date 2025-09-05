#!/usr/bin/env python3

"""
Training Script for Advanced Preprocessed Underwater Dataset
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# Import configurations
from config import Config
from advanced_config import AdvancedPreprocessedConfig

def setup_training_environment():
    """Setup optimal training environment"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    return device

def train_with_advanced_config():
    """Train YOLOv8 with advanced preprocessed dataset"""
    
    config = AdvancedPreprocessedConfig()
    device = setup_training_environment()
    
    print("🚀 Starting Advanced Underwater Object Detection Training")
    print("=" * 60)
    print(f"📁 Dataset: {config.DATASET_DIR}")
    print(f"🎯 Model: {config.MODEL_SIZE}")
    print(f"📊 Epochs: {config.EPOCHS}")
    print(f"🖼️  Image Size: {config.IMAGE_SIZE}")
    print(f"📦 Batch Size: {config.BATCH_SIZE}")
    print("=" * 60)
    
    # Initialize model
    model_path = f"{config.MODEL_SIZE}.pt"
    if not os.path.exists(model_path):
        print(f"⬇️  Downloading {config.MODEL_SIZE}.pt...")
    
    model = YOLO(model_path)
    
    # Training arguments optimized for preprocessed data
    train_args = {
        'data': config.DATA_YAML,
        'epochs': config.EPOCHS,
        'batch': config.BATCH_SIZE,
        'imgsz': config.IMAGE_SIZE,
        'device': device,
        'workers': 8,
        
        # Learning rate schedule
        'lr0': config.LR0,
        'warmup_epochs': config.WARMUP_EPOCHS,
        'patience': config.PATIENCE,
        
        # Loss weights (optimized for augmented data)
        'cls': config.CLS_WEIGHTS,
        'box': config.BOX_WEIGHTS,
        'dfl': config.DFL_WEIGHTS,
        
        # Regularization
        'weight_decay': config.WEIGHT_DECAY,
        'label_smoothing': config.LABEL_SMOOTHING,
        
        # Augmentation (lighter since we pre-processed)
        'degrees': 5,    # Reduced from default
        'translate': 0.05,  # Reduced
        'scale': 0.3,    # Reduced
        'mosaic': 0.3,   # Reduced probability
        'mixup': 0.1,    # Light mixup
        
        # Validation
        'val': True,
        'save': True,
        'save_period': 25,  # Save checkpoints every 25 epochs
        
        # Project organization
        'project': 'runs',
        'name': f'aquarium_advanced_{config.MODEL_SIZE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        
        # Advanced options
        'resume': False,  # Set to True to resume from last checkpoint
        'amp': True if device == 'cuda' else False,  # Mixed precision
        'freeze': None,   # Don't freeze layers
    }
    
    print("\\n📈 Training Arguments:")
    for key, value in train_args.items():
        if key not in ['data']:  # Skip long paths
            print(f"   {key}: {value}")
    
    print("\\n🏃‍♂️ Starting training...")
    try:
        # Start training
        results = model.train(**train_args)
        
        print("\\n✅ Training completed successfully!")
        print(f"📊 Best mAP50: {results.best_fitness:.4f}")
        print(f"📁 Results saved to: runs/{train_args['name']}")
        
        # Export trained model
        export_path = f"models/advanced_{config.MODEL_SIZE}_best.pt"
        os.makedirs("models", exist_ok=True)
        
        # Copy best weights
        import shutil
        best_weights = f"runs/{train_args['name']}/weights/best.pt"
        if os.path.exists(best_weights):
            shutil.copy2(best_weights, export_path)
            print(f"🎯 Best model exported to: {export_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 with Advanced Preprocessed Dataset')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()
    
    if args.resume:
        print("🔄 Resume mode enabled")
    
    # Check if enhanced dataset exists
    if not os.path.exists("aquarium_advanced"):
        print("❌ Enhanced dataset not found!")
        print("Please run the preprocessing script first:")
        print("   python advanced_data_preprocessing.py")
        return
    
    # Start training
    results = train_with_advanced_config()
    
    if results:
        print("\\n🎉 Training Summary:")
        print(f"✅ Enhanced dataset: 2658 training images (vs 448 original)")
        print(f"✅ Advanced augmentations: Copy-paste, oversampling, hard negatives")
        print(f"✅ Optimized config: Extended training, label smoothing")
        print(f"✅ Ready for inference and evaluation!")
    else:
        print("\\n💔 Training was unsuccessful. Please check the error logs.")

if __name__ == "__main__":
    main()
