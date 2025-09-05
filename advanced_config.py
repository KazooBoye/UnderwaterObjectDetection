
"""
Optimized Training Configuration for Advanced Preprocessed Dataset
"""

from config import Config
import torch

class AdvancedPreprocessedConfig(Config):
    """Configuration optimized for advanced preprocessed underwater dataset"""
    
    # Dataset paths
    DATASET_DIR = "aquarium_advanced"
    DATA_YAML = "aquarium_advanced/data.yaml"
    
    # Model configuration
    MODEL_SIZE = "yolov8m"  # Medium model for balance of speed/accuracy
    
    # Extended training for complex augmented dataset
    EPOCHS = 300  # More epochs needed for complex augmented data
    BATCH_SIZE = 12  # Slightly smaller for complex augmentations
    IMAGE_SIZE = 800  # Higher resolution for small objects
    
    # Learning rate schedule - more conservative for augmented data
    LR0 = 0.0005  # Lower initial rate
    WARMUP_EPOCHS = 15  # Extended warmup
    PATIENCE = 75  # More patience for complex dataset
    
    # Loss function optimization
    CLS_WEIGHTS = 0.8   # Slightly lower for oversampled data
    BOX_WEIGHTS = 12.0  # Higher for precise localization
    DFL_WEIGHTS = 2.5   # Higher for small object distribution
    
    # Label smoothing for overconfident predictions
    LABEL_SMOOTHING = 0.1
    
    # Reduced additional augmentation (already preprocessed)
    HSV_H = 0.005  # Minimal since already color corrected
    HSV_S = 0.2    # Reduced
    HSV_V = 0.1    # Reduced
    DEGREES = 2.0  # Minimal rotation
    TRANSLATE = 0.02  # Minimal translation
    SCALE = 0.1    # Minimal scaling
    
    # Mosaic and MixUp (let YOLOv8 handle these)
    MOSAIC = 0.8   # Keep mosaic for multi-object learning
    MIXUP = 0.1    # Light mixup
    COPY_PASTE = 0.0  # Disabled (already done in preprocessing)
    
    # Class-weighted loss for remaining imbalance
    CLASS_WEIGHTS = True  # Enable automatic class weighting
    
    @classmethod
    def get_training_args(cls):
        """Get training arguments optimized for preprocessed data"""
        args = super().get_training_args()
        
        # Override with preprocessed-specific settings
        args.update({
            'data': cls.DATA_YAML,
            'epochs': cls.EPOCHS,
            'batch': cls.BATCH_SIZE,
            'imgsz': cls.IMAGE_SIZE,
            'lr0': cls.LR0,
            'warmup_epochs': cls.WARMUP_EPOCHS,
            'patience': cls.PATIENCE,
            'box': cls.BOX_WEIGHTS,
            'cls': cls.CLS_WEIGHTS,
            'dfl': cls.DFL_WEIGHTS,
            'hsv_h': cls.HSV_H,
            'hsv_s': cls.HSV_S,
            'hsv_v': cls.HSV_V,
            'degrees': cls.DEGREES,
            'translate': cls.TRANSLATE,
            'scale': cls.SCALE,
            'mosaic': cls.MOSAIC,
            'mixup': cls.MIXUP,
            'copy_paste': cls.COPY_PASTE,
            'label_smoothing': cls.LABEL_SMOOTHING,
            'multi_scale': True,  # Multi-scale training
            'save_period': 20,    # Save more frequently
            'project': 'runs/advanced_preprocessed',
            'name': 'yolov8_advanced',
            'cos_lr': True,       # Cosine learning rate
            'close_mosaic': 30,   # Close mosaic earlier
        })
        
        return args

# Usage
if __name__ == "__main__":
    config = AdvancedPreprocessedConfig()
    print("Advanced preprocessed training configuration loaded!")
