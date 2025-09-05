# YOLOv8 Underwater Object Detection - Configuration Template
# Copy this to config.py and customize for your environment
# This file should be added to .gitignore

import os
from pathlib import Path

# Try to import torch, but make it optional for cases where it's not installed
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'
    print("⚠️ PyTorch not found, defaulting to CPU")

class Config:
    """Configuration class for YOLOv8 underwater object detection project"""
    
    # Auto-detect project root (recommended - no manual changes needed)
    PROJECT_ROOT = Path(__file__).parent.absolute()
    
    # Manual override (uncomment and modify if auto-detection fails)
    # PROJECT_ROOT = Path("/your/custom/path/to/UnderwaterObjectDetection")
    
    # Dataset paths (relative to project root)
    DATASET_DIR = PROJECT_ROOT / "aquarium_light_aug"  # Using light augmented dataset
    DATA_YAML = DATASET_DIR / "data.yaml"
    
    # Output directories
    RUNS_DIR = PROJECT_ROOT / "runs"
    WEIGHTS_DIR = RUNS_DIR / "weights"
    LOGS_DIR = RUNS_DIR / "logs"
    
    # Model Configuration
    MODEL_SIZE = "yolov8m.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    EXPERIMENT_NAME = "aquarium_yolov8l"
    
    # System-specific settings
    BATCH_SIZE = 8  # Adjust based on GPU memory
    WORKERS = min(8, os.cpu_count() or 1)
    DEVICE = DEVICE  # Set above with torch detection
    
    # Training Hyperparameters
    EPOCHS = 200
    PATIENCE = 40  # Early stopping patience
    IMAGE_SIZE = 640
    
    # Learning Rate Configuration
    LR0 = 0.005  # Initial learning rate
    LRF = 0.1    # Final learning rate factor
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 3
    WARMUP_MOMENTUM = 0.8
    WARMUP_BIAS_LR = 0.1
    
    # Loss Function Weights (adjust for class imbalance)
    CLS_LOSS_WEIGHT = 0.5   # Reduced for imbalanced dataset
    BOX_LOSS_WEIGHT = 7.5   # Standard box regression loss
    DFL_LOSS_WEIGHT = 1.5   # Distribution focal loss
    
    # Data Augmentation Settings (minimal for pre-augmented datasets)
    HSV_H = 0.0      # Hue augmentation
    HSV_S = 0.0      # Saturation augmentation  
    HSV_V = 0.0      # Value/brightness augmentation
    DEGREES = 0      # Rotation degrees
    TRANSLATE = 0.0  # Translation
    SCALE = 0.0      # Scale augmentation
    SHEAR = 0        # Shear augmentation
    PERSPECTIVE = 0  # Perspective augmentation
    FLIPUD = 0       # Vertical flip probability
    FLIPLR = 0       # Horizontal flip probability
    MOSAIC = 0       # Mosaic augmentation probability
    MIXUP = 0        # Mixup augmentation probability
    COPY_PASTE = 0   # Copy-paste augmentation
    
    # Training Optimization
    OPTIMIZER = 'Adam'  # Options: 'SGD', 'Adam', 'AdamW'
    CLOSE_MOSAIC = 0   # Epoch to close mosaic augmentation
    AMP = True         # Automatic Mixed Precision
    FRACTION = 1.0     # Dataset fraction to use
    
    # Validation and Monitoring
    VAL = True
    PLOTS = True
    SAVE = True
    SAVE_PERIOD = 20   # Save checkpoint every N epochs
    CACHE = False      # Cache dataset in memory
    VERBOSE = True
    
    # Legacy path mapping for migration (will be removed in future versions)
    LEGACY_PATHS = {
        "macos_root": "/Users/username/Desktop/Coding/UnderwaterObjectDetection",
        "linux_root": str(PROJECT_ROOT)
    }
    
    @classmethod
    def get_data_yaml_path(cls):
        """Get the data.yaml path as string (for YOLO training)"""
        return str(cls.DATA_YAML)
    
    @classmethod
    def get_runs_dir(cls):
        """Get the runs directory path as string"""
        cls.RUNS_DIR.mkdir(exist_ok=True)
        return str(cls.RUNS_DIR)
    
    @classmethod  
    def get_project_root(cls):
        """Get the project root as string"""
        return str(cls.PROJECT_ROOT)
    
    @classmethod
    def get_training_args(cls):
        """Get complete training arguments dictionary for YOLO"""
        return {
            # Basic training settings
            'data': cls.get_data_yaml_path(),
            'epochs': cls.EPOCHS,
            'patience': cls.PATIENCE,
            'batch': cls.BATCH_SIZE,
            'imgsz': cls.IMAGE_SIZE,
            'device': cls.DEVICE,
            'workers': cls.WORKERS,
            
            # Learning rate and optimization
            'lr0': cls.LR0,
            'lrf': cls.LRF,
            'momentum': cls.MOMENTUM,
            'weight_decay': cls.WEIGHT_DECAY,
            'warmup_epochs': cls.WARMUP_EPOCHS,
            'warmup_momentum': cls.WARMUP_MOMENTUM,
            'warmup_bias_lr': cls.WARMUP_BIAS_LR,
            
            # Loss function weights
            'cls': cls.CLS_LOSS_WEIGHT,
            'box': cls.BOX_LOSS_WEIGHT,
            'dfl': cls.DFL_LOSS_WEIGHT,
            
            # Data augmentation
            'hsv_h': cls.HSV_H,
            'hsv_s': cls.HSV_S,
            'hsv_v': cls.HSV_V,
            'degrees': cls.DEGREES,
            'translate': cls.TRANSLATE,
            'scale': cls.SCALE,
            'shear': cls.SHEAR,
            'perspective': cls.PERSPECTIVE,
            'flipud': cls.FLIPUD,
            'fliplr': cls.FLIPLR,
            'mosaic': cls.MOSAIC,
            'mixup': cls.MIXUP,
            'copy_paste': cls.COPY_PASTE,
            
            # Training optimization
            'optimizer': cls.OPTIMIZER,
            'close_mosaic': cls.CLOSE_MOSAIC,
            'amp': cls.AMP,
            'fraction': cls.FRACTION,
            
            # Validation and monitoring
            'val': cls.VAL,
            'plots': cls.PLOTS,
            'save': cls.SAVE,
            'save_period': cls.SAVE_PERIOD,
            'cache': cls.CACHE,
            'project': cls.get_runs_dir(),
            'name': cls.EXPERIMENT_NAME,
            'exist_ok': True,
            'pretrained': True,
            'verbose': cls.VERBOSE,
        }
