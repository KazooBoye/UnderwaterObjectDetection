#!/usr/bin/env python3
"""
YOLOv8 Training Script - Configuration Driven
Uses all parameters from config.py for consistent, centralized configuration
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Import configuration
try:
    from config import Config
    print("✅ Config file loaded successfully")
except ImportError:
    print("❌ Config file not found. Creating from template...")
    import shutil
    shutil.copy("config_template.py", "config.py")
    from config import Config
    print("✅ Created config.py from template")

def print_config_summary():
    """Print current configuration summary"""
    print("\n" + "="*60)
    print("🔧 TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"📁 Dataset: {Config.get_data_yaml_path()}")
    print(f"🎯 Model: {Config.MODEL_SIZE}")
    print(f"🏷️ Experiment: {Config.EXPERIMENT_NAME}")
    print(f"🖥️ Device: {Config.DEVICE}")
    print(f"📊 Batch Size: {Config.BATCH_SIZE}")
    print(f"⏱️ Epochs: {Config.EPOCHS} (patience: {Config.PATIENCE})")
    print(f"📈 Learning Rate: {Config.LR0} → {Config.LR0 * Config.LRF}")
    print(f"🎨 Augmentation: {'DISABLED' if Config.MOSAIC == 0 else 'ENABLED'}")
    print(f"⚖️ Loss Weights: cls={Config.CLS_LOSS_WEIGHT}, box={Config.BOX_LOSS_WEIGHT}")
    print(f"🔄 Optimizer: {Config.OPTIMIZER}")
    print("="*60)

def train_yolo_configured():
    """
    Train YOLOv8 using all parameters from config.py
    No hardcoded parameters - everything comes from Config class
    """
    
    print("🚀 YOLO TRAINING - CONFIGURATION DRIVEN")
    
    # Print configuration summary
    print_config_summary()
    
    # Initialize YOLO model using config
    model = YOLO(Config.MODEL_SIZE)
    print(f"\n✅ Model initialized: {Config.MODEL_SIZE}")
    
    # Get training arguments from config - no local overrides!
    training_args = Config.get_training_args()
    
    print(f"\n🔧 Training arguments loaded from config.py")
    print(f"📝 Total parameters: {len(training_args)}")
    
    # Verify critical paths exist
    data_yaml_path = Path(training_args['data'])
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"❌ Dataset YAML not found: {data_yaml_path}")
    
    print(f"\n🎯 Starting training with configuration:")
    print(f"   - Model: {Config.MODEL_SIZE}")
    print(f"   - Dataset: {Config.DATASET_DIR.name}")
    print(f"   - Epochs: {Config.EPOCHS}")
    print(f"   - Batch: {Config.BATCH_SIZE}")
    print(f"   - Augmentation: {'OFF' if Config.MOSAIC == 0 else 'ON'}")
    
    try:
        # Train using config parameters
        results = model.train(**training_args)
        
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"📁 Results: {results.save_dir}")
        print(f"🎖️ Best weights: {results.save_dir}/weights/best.pt")
        print(f"📊 Logs: {results.save_dir}/results.csv")
        
        return results
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED")
        print(f"Error: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"   1. Check if dataset path is correct: {Config.DATASET_DIR}")
        print(f"   2. Verify GPU memory for batch size: {Config.BATCH_SIZE}")
        print(f"   3. Ensure data.yaml exists: {Config.DATA_YAML}")
        raise

def main():
    """Main training function"""
    
    # Verify configuration
    if not Config.DATA_YAML.exists():
        print(f"❌ Dataset YAML not found: {Config.DATA_YAML}")
        print(f"💡 Available datasets:")
        for dataset_dir in Config.PROJECT_ROOT.glob("aquarium*"):
            if dataset_dir.is_dir():
                yaml_file = dataset_dir / "data.yaml"
                status = "✅" if yaml_file.exists() else "❌"
                print(f"   {status} {dataset_dir.name}")
        
        print(f"\n🔧 Update Config.DATASET_DIR in config.py to point to the correct dataset")
        return
    
    # Start training
    train_yolo_configured()

if __name__ == "__main__":
    main()
