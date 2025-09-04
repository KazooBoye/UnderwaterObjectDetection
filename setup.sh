#!/bin/bash
# Setup script for YOLOv8 Aquarium Object Detection Training

echo "=== YOLOv8 Aquarium Dataset Training Setup ==="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - optimizing for Apple Silicon/Intel"
fi

# Create virtual environment (recommended)
echo "1. Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Virtual environment created and activated."
echo ""

# Upgrade pip
echo "2. Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install requirements
echo "3. Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "4. Verifying installations..."

# Test imports
python -c "
try:
    import torch
    print(f'✓ PyTorch {torch.__version__} installed')
    if torch.cuda.is_available():
        print(f'✓ CUDA available: {torch.cuda.get_device_name()}')
    elif torch.backends.mps.is_available():
        print('✓ MPS (Apple Silicon GPU) available')
    else:
        print('! GPU not available - training will use CPU (slower)')
except ImportError:
    print('✗ PyTorch installation failed')

try:
    from ultralytics import YOLO
    print('✓ Ultralytics YOLOv8 installed')
except ImportError:
    print('✗ Ultralytics installation failed')

try:
    import cv2
    print(f'✓ OpenCV {cv2.__version__} installed')
except ImportError:
    print('✗ OpenCV installation failed')

try:
    import numpy as np
    print(f'✓ NumPy {np.__version__} installed')
except ImportError:
    print('✗ NumPy installation failed')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run data preprocessing: python preprocess_data.py"
echo "3. Start training: python train_yolov8.py"
echo ""
echo "For faster experimentation, start with the balanced subset option in preprocessing."
echo ""

# Make scripts executable
chmod +x preprocess_data.py
chmod +x train_yolov8.py
chmod +x setup_training.py

echo "Scripts made executable."
echo ""
echo "Happy training! 🚀"
