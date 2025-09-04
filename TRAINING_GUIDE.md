# YOLOv8 Aquarium Object Detection Training Guide

## Dataset Analysis Summary

Based on your comprehensive dataset analysis, the main challenges are:

### 🚨 Critical Issues Identified:
1. **Severe Class Imbalance**: Fish (59%) vs Starfish (2.3%) = 25:1 ratio
2. **Multi-scale Objects**: Size variations from 0.6% to 6% normalized area  
3. **Complex Multi-object Scenes**: Average 7.4 objects per image
4. **Distribution Drift**: Different class ratios across train/test splits

### 📊 Class Distribution:
- **Dominant**: Fish (1,961 samples, 59%)
- **Medium**: Jellyfish (384 samples, 11.6%), Shark (921 samples, 27.8%)
- **Minority**: Starfish (78 samples, 2.3%), Stingray (136 samples, 4.1%), Puffin (175 samples, 5.3%)

## Preprocessing Steps Required

### 1. Environment Setup
```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

### 2. Data Preprocessing (Choose One Approach)

#### Option A: Balanced Subset (Recommended for Experimentation) 
```bash
python preprocess_data.py
# Choose option 1: Creates a smaller, balanced dataset (~150 samples per class)
```

#### Option B: Full Dataset with Augmentation
```bash
python preprocess_data.py  
# Choose option 2: Augments minority classes in full dataset
```

#### Option C: Both Approaches
```bash
python preprocess_data.py
# Choose option 3: Creates both balanced subset and augmented full dataset
```

### 3. Training Configuration

The training script implements these optimizations:

#### 🎯 **Class Imbalance Solutions**:
- **Extended Training**: 300 epochs (vs standard 100) for minority class learning
- **Focal Loss Components**: Weighted cls/box/dfl losses
- **Advanced Augmentation**: Mosaic (1.0) + Mixup (0.15) + Copy-paste (0.3)
- **SGD Optimizer**: Better stability for imbalanced data

#### 📏 **Multi-scale Detection**:
- **Aggressive Scale Augmentation**: scale=0.5 (handles 0.6%-6% area variation)
- **Multi-resolution Training**: imgsz=640 with scale variations

#### 🔧 **Multi-object Scene Handling**:
- **Mosaic Augmentation**: Combines 4 images for complex scene training
- **High max_det**: Up to 300 detections per image
- **Optimized NMS**: IoU=0.5 for dense object scenarios

## Training Process

### 1. Start Training
```bash
# Activate virtual environment
source venv/bin/activate

# Start training
python train_yolov8.py
```

### 2. Monitor Training
The script will output:
- Training progress with loss curves
- Validation metrics every epoch  
- Per-class performance (crucial for imbalanced data)
- Model checkpoints every 50 epochs

### 3. Key Metrics to Monitor

#### ❌ **Don't rely on**:
- Overall accuracy (misleading due to class imbalance)
- Standard mAP without per-class breakdown

#### ✅ **Focus on**:
- **Per-class mAP50**: Especially for minority classes (starfish, stingray, puffin)
- **Class-specific Precision/Recall**: Balance between false positives and missed detections
- **Confusion Matrix**: Identify which classes are confused with each other
- **Loss Convergence**: Ensure all loss components (box, cls, dfl) are decreasing

### 4. Expected Training Outcomes

#### **Success Indicators**:
- Minority classes achieve >0.3 mAP50 (given limited data)
- No single class dominates predictions (avoid "always predict fish")
- Loss curves converge smoothly without excessive overfitting
- Validation performance doesn't degrade significantly vs training

#### **Warning Signs**:
- Minority classes show 0.0 mAP50 (model ignoring them)
- Validation loss increases while training loss decreases (overfitting)
- Extremely high precision but low recall for minority classes

## Advanced Optimization Strategies

### If Initial Results Are Poor:

#### 1. **Loss Function Adjustments**:
```python
# In train_yolov8.py, modify these values:
'cls': 2.0,      # Increase classification loss weight
'box': 5.0,      # Decrease box loss weight  
'dfl': 1.0,      # Adjust distribution focal loss
```

#### 2. **Learning Rate Scheduling**:
```python
'lr0': 0.001,    # Lower initial learning rate
'lrf': 0.0001,   # Lower final learning rate
'warmup_epochs': 5,  # More warmup epochs
```

#### 3. **Enhanced Data Augmentation**:
```python
'mixup': 0.3,        # Increase mixup (better for imbalance)
'copy_paste': 0.5,   # Increase copy-paste 
'scale': 0.7,        # More aggressive scaling
```

#### 4. **Ensemble Methods**:
Train multiple models with different random seeds and ensemble predictions.

### Model Size Considerations:

#### **Start with YOLOv8n** (nano):
- Faster training and inference
- Good for experimentation and prototyping
- Sufficient for proof-of-concept

#### **Upgrade to YOLOv8s/m** if needed:
- Better accuracy for complex scenes  
- More parameters for learning rare classes
- Longer training time

## Evaluation Protocol

### 1. **Per-Class Analysis**:
```bash
# The evaluation script automatically provides per-class metrics
# Focus on minority class performance:
# - Starfish mAP50 > 0.2 (given only 78 samples)
# - Stingray mAP50 > 0.3 (given 136 samples)  
# - Puffin mAP50 > 0.3 (given 175 samples)
```

### 2. **Threshold Optimization**:
```python
# Test different confidence thresholds for each class
# Minority classes may need lower thresholds (0.1-0.3)
# Majority classes can use higher thresholds (0.4-0.6)
```

### 3. **Real-world Testing**:
- Test on unseen underwater images
- Verify performance across different lighting conditions
- Check for bias toward dominant classes

## Expected Timeline

- **Environment Setup**: 15-30 minutes
- **Data Preprocessing**: 30-60 minutes (depending on approach)
- **Training (YOLOv8n)**: 
  - GPU: 4-8 hours for 300 epochs
  - CPU: 24-48 hours for 300 epochs
- **Evaluation**: 15-30 minutes

## Troubleshooting Common Issues

### 1. **CUDA Out of Memory**:
```python
# Reduce batch size in train_yolov8.py:
'batch': 8,  # Instead of 16
```

### 2. **Poor Minority Class Performance**:
- Try balanced subset first for faster iteration
- Increase augmentation for minority classes
- Consider synthetic data generation
- Use ensemble of models trained on different class balances

### 3. **Overfitting**:
```python
'dropout': 0.2,        # Increase dropout
'weight_decay': 0.001, # Increase regularization  
```

### 4. **Slow Training**:
- Use GPU if available (CUDA or MPS for Apple Silicon)
- Reduce image size temporarily: `'imgsz': 416`
- Use smaller model initially: YOLOv8n instead of YOLOv8s

## Next Steps After Training

1. **Analyze Results**: Focus on per-class confusion matrix
2. **Threshold Tuning**: Optimize confidence thresholds per class  
3. **Error Analysis**: Identify failure cases and patterns
4. **Data Collection**: If possible, collect more samples of minority classes
5. **Deployment**: Convert model for inference (ONNX, TensorRT, etc.)

---

**Remember**: This is a challenging dataset due to extreme class imbalance. Success should be measured by balanced performance across all classes, not just overall accuracy. The minority classes (starfish, stingray, puffin) are the real test of model quality.
