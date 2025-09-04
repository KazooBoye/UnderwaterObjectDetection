# YOLOv8 Underwater Object Detection - Configuration Template
# Copy this to config.py and customize for your environment
# This file should be added to .gitignore

import os
from pathlib import Path

class Config:
    """Configuration class for YOLOv8 underwater object detection project"""
    
    # Auto-detect project root (recommended - no manual changes needed)
    PROJECT_ROOT = Path(__file__).parent.absolute()
    
    # Manual override (uncomment and modify if auto-detection fails)
    # PROJECT_ROOT = Path("/your/custom/path/to/UnderwaterObjectDetection")
    
    # Dataset paths (relative to project root)
    DATASET_DIR = PROJECT_ROOT / "aquarium_pretrain"
    DATA_YAML = DATASET_DIR / "data.yaml"
    
    # Output directories
    RUNS_DIR = PROJECT_ROOT / "runs"
    WEIGHTS_DIR = RUNS_DIR / "weights"
    LOGS_DIR = RUNS_DIR / "logs"
    
    # Training configuration
    EXPERIMENT_NAME = "aquarium_yolov8_balanced"
    MODEL_SIZE = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # System-specific settings
    BATCH_SIZE = 16
    WORKERS = min(8, os.cpu_count() or 1)
    
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
