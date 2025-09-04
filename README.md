# YOLOv8 Underwater Object Detection

A YOLOv8-based deep learning system for detecting underwater creatures with severe class imbalance optimization.

## Project Overview

**Objective**: Train YOLOv8 to detect 7 underwater creatures in aquarium environments
- **Classes**: Fish, Jellyfish, Penguin, Puffin, Shark, Starfish, Stingray
- **Main Challenge**: Severe class imbalance (Fish: 59%, Starfish: 2.3%)
- **Approach**: Extended training, focal loss, advanced augmentation

## Quick Start

### Single Setup Command
```bash
python3 init_config.py
```
This single script will:
- Initialize your personal configuration
- Detect your system (conda/venv)
- Set up the complete environment
- Make everything ready for training
### Start Training
```bash
# Preprocess data (choose balanced subset for experimentation)
python3 preprocess_data.py

# Start training (300 epochs, optimized for class imbalance)
python3 train_yolov8.py
```

## Project Structure

```
UnderwaterObjectDetection/
├── datasetAnalysis.ipynb     # Dataset analysis notebook
├── config_template.py       # Configuration template
├── config.py               # Personal config (auto-generated, gitignored)
├── init_config.py          # Single setup script (config + environment)
├── preprocess_data.py      # Data preprocessing
├── train_yolov8.py         # Main training script
├── environment.yml         # Conda environment definition
└── aquarium_pretrain/         # Dataset directory
    ├── data.yaml              # Dataset config (auto-generated, gitignored)
    ├── data_template.yaml     # Dataset template
    ├── train/, valid/, test/   # Dataset splits
    └── ...
```

## Dataset Challenges & Solutions

### Critical Issues
1. **Severe Class Imbalance**: Fish (59%) vs Starfish (2.3%) = 25:1 ratio
2. **Multi-scale Objects**: Size variations from 0.6% to 6% normalized area  
3. **Complex Multi-object Scenes**: Average 7.4 objects per image
4. **Distribution Drift**: Different class ratios across train/test splits

### Optimizations Applied
- **Extended Training**: 300 epochs (vs standard 100)
- **Focal Loss Components**: Weighted cls/box/dfl losses
- **Advanced Augmentation**: Mosaic (1.0) + Mixup (0.15) + Copy-paste (0.3)
- **Multi-scale Training**: Aggressive scale augmentation (scale=0.5)
- **SGD Optimizer**: Better stability for imbalanced data

## Configuration System

### Privacy-Safe Design
- **`config_template.py`**: Safe template (commit this)
- **`config.py`**: Personal paths (gitignored)  
- **Auto-detection**: Dynamic path resolution
- **Cross-platform**: Works on any system

### Customizable Settings
```python
# In config.py
class Config:
    MODEL_SIZE = "yolov8n.pt"    # yolov8n/s/m/l/x
    BATCH_SIZE = 16              # Adjust for your GPU
    WORKERS = 8                  # Auto-detected
    EXPERIMENT_NAME = "aquarium_yolov8_balanced"
```

## Environment Setup

### System Requirements
- **Minimum**: Linux/macOS, Python 3.8+, 8GB RAM
- **Recommended**: NVIDIA GPU, 16GB+ RAM, CUDA support
- **Optimal**: CUDA GPU with 8GB+ VRAM, SSD storage

### Setup Options

The `init_config.py` script automatically detects your system and offers:

#### Option 1: Anaconda/Miniconda (Recommended)
- Better optimized scientific packages
- Superior CUDA integration
- Isolated environment
- Automatic installation if not present

#### Option 2: Python venv (Fallback)  
- Lightweight alternative
- Uses system Python
- Good for CPU-only setups

#### Option 3: Manual Setup
```bash
# If you prefer manual control
conda env create -f environment.yml
conda activate yolov8_aquarium
```

## Training Process

### 1. Data Preprocessing Options
```bash
python3 preprocess_data.py
```
- **Option 1**: Balanced subset (~150 samples/class) - for experimentation
- **Option 2**: Full dataset with augmentation - for production
- **Option 3**: Both approaches - comprehensive

### 2. Training Configuration
The training script implements specialized optimizations:

#### Class Imbalance Solutions
- Extended 300-epoch training schedule
- Weighted loss functions (cls/box/dfl)
- Advanced data augmentation pipeline
- Focal loss mechanisms

#### Multi-scale Detection  
- Aggressive scale augmentation (scale=0.5)
- Multi-resolution training (imgsz=640)
- Handles 0.6%-6% object area variation

#### Multi-object Scene Handling
- Mosaic augmentation (combines 4 images)
- High detection limit (max_det=300)
- Optimized NMS (IoU=0.5)

### 3. Monitoring Training
Key metrics to watch:
- **Per-class mAP50**: Focus on minority classes
- **Confusion Matrix**: Check for class bias
- **Loss Convergence**: All components decreasing
- **Validation Performance**: Avoid overfitting

### 4. Success Indicators
- All minority classes achieve >0.3 mAP50
- No single class dominates predictions
- Smooth loss convergence
- Balanced confusion matrix

## Performance Optimization

### GPU Optimization
```python
# Automatic GPU detection
'device': 0 if torch.cuda.is_available() else 'cpu'

# Memory management
'batch': 16,     # Reduce if GPU memory limited
'workers': 8,    # Adjust for CPU cores
```

### CPU Optimization  
```python
# Dynamic worker allocation
'workers': min(8, os.cpu_count() or 1)

# Conservative memory usage
'cache': False,  # No dataset caching
```

### Training Time Estimates
- **GPU (CUDA)**: 4-8 hours (300 epochs)
- **CPU Only**: 24-48 hours (300 epochs)
- **Apple Silicon**: 6-12 hours (300 epochs)

## Model Evaluation

### Evaluation Protocol
1. **Per-class Analysis**: Focus on minority performance
2. **Threshold Optimization**: Different thresholds per class  
3. **Error Analysis**: Identify failure patterns
4. **Real-world Testing**: Various lighting conditions

### Expected Performance
- **Fish**: >0.7 mAP50 (majority class)
- **Shark/Jellyfish**: >0.5 mAP50 (medium classes)
- **Starfish/Stingray/Puffin**: >0.3 mAP50 (minorities)

## Troubleshooting

### Common Issues

#### Environment Setup
```bash
# Conda not found
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Python version issues
python3 --version  # Ensure Python 3.8+
```

#### CUDA Problems
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Memory Issues
```python
# Reduce batch size in config.py
BATCH_SIZE = 8  # Instead of 16

# Reduce workers
WORKERS = 2     # Instead of auto-detected
```

#### Training Problems
- **Poor minority performance**: Use balanced subset first
- **Overfitting**: Increase dropout, reduce epochs
- **Slow training**: Check GPU utilization, reduce image size

## Advanced Usage

### Custom Model Sizes
```python
# In config.py
MODEL_SIZE = "yolov8s.pt"  # Larger model for better accuracy
BATCH_SIZE = 8             # Reduce batch size accordingly
```

### Hyperparameter Tuning
```python
# In train_yolov8.py, modify:
'lr0': 0.001,      # Lower learning rate
'epochs': 500,     # More training
'mixup': 0.3,      # More augmentation
```

### Ensemble Methods
Train multiple models with different configurations and ensemble predictions for better performance.

## Deployment

### Model Export
```python
# After training, convert model:
from ultralytics import YOLO
model = YOLO('runs/aquarium_yolov8_balanced/weights/best.pt')
model.export(format='onnx')    # For cross-platform inference
model.export(format='engine')  # For TensorRT optimization
```

### Inference Pipeline
```python
# Load trained model
model = YOLO('path/to/best.pt')

# Predict on new images
results = model('path/to/image.jpg', conf=0.25)
results[0].show()  # Display results
```

## Contributing

1. **Fork** the repository
2. **Initialize** your config: `python3 init_config.py`
3. **Setup** environment: `./setup_complete.sh`
4. **Make** your changes
5. **Test** with your dataset
6. **Submit** pull request

## License

This project is open source. Please check the license file for details.

## Support

### Quick Diagnostics
```bash
# Check environment
conda list | head -10          # Conda packages
python3 -c "import torch; print(torch.__version__)"  # PyTorch

# Check GPU
nvidia-smi                     # GPU status
python3 -c "import torch; print(torch.cuda.is_available())"  # CUDA

# Check config
python3 -c "from config import Config; print(Config.get_project_root())"
```

### Getting Help
1. **Check configuration**: Ensure `config.py` exists and is valid
2. **Verify environment**: All packages installed correctly  
3. **Monitor resources**: `htop` for CPU/memory, `nvidia-smi` for GPU
4. **Review logs**: Check `runs/` directory for training outputs

---

**Ready to detect underwater creatures with state-of-the-art YOLOv8!**

This project demonstrates advanced techniques for handling severe class imbalance in object detection, making it valuable for both research and practical applications.
