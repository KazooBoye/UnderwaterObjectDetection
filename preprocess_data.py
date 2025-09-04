#!/usr/bin/env python3
"""
Enhanced Data Augmentation Script for Minority Classes
Addresses severe class imbalance in aquarium dataset
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import random
from collections import Counter

class MinorityClassAugmenter:
    """Augment minority classes to balance the dataset"""
    
    def __init__(self, data_root, target_samples_per_class=500):
        self.data_root = Path(data_root)
        self.target_samples = target_samples_per_class
        self.class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        
    def analyze_class_distribution(self):
        """Analyze current class distribution"""
        train_labels = self.data_root / "train" / "labels"
        class_counts = Counter()
        
        for label_file in train_labels.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        
        print("Current class distribution:")
        for class_id in range(7):
            count = class_counts.get(class_id, 0)
            needed = max(0, self.target_samples - count)
            print(f"  {self.class_names[class_id]}: {count} samples (need {needed} more)")
        
        return class_counts
    
    def get_images_with_class(self, class_id):
        """Get all images containing a specific class"""
        train_labels = self.data_root / "train" / "labels" 
        images_with_class = []
        
        for label_file in train_labels.glob("*.txt"):
            has_class = False
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and int(parts[0]) == class_id:
                        has_class = True
                        break
            
            if has_class:
                img_file = self.data_root / "train" / "images" / f"{label_file.stem}.jpg"
                if img_file.exists():
                    images_with_class.append((img_file, label_file))
        
        return images_with_class
    
    def augment_image(self, image_path, output_path, augmentation_type):
        """Apply specific augmentation to an image"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        height, width = img.shape[:2]
        
        if augmentation_type == 'brightness':
            # Adjust brightness (important for underwater scenes)
            alpha = random.uniform(0.7, 1.3)  # Brightness factor
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            
        elif augmentation_type == 'contrast':
            # Adjust contrast
            alpha = random.uniform(0.8, 1.2)  # Contrast factor
            beta = random.uniform(-10, 10)    # Brightness offset
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
        elif augmentation_type == 'blur':
            # Apply slight blur (water effect)
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
        elif augmentation_type == 'noise':
            # Add slight noise
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
        elif augmentation_type == 'hue_shift':
            # Shift hue (underwater color variations)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue_shift = random.randint(-10, 10)
            hsv[:,:,0] = cv2.add(hsv[:,:,0], hue_shift)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif augmentation_type == 'horizontal_flip':
            # Horizontal flip
            img = cv2.flip(img, 1)
            
        # Save augmented image
        cv2.imwrite(str(output_path), img)
        return output_path
    
    def augment_labels(self, label_path, output_path, augmentation_type):
        """Adjust labels for augmented images"""
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        adjusted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = parts
                x_center, y_center = float(x_center), float(y_center)
                
                # Adjust coordinates for horizontal flip
                if augmentation_type == 'horizontal_flip':
                    x_center = 1.0 - x_center  # Flip x coordinate
                
                adjusted_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
            else:
                adjusted_lines.append(line)
        
        with open(output_path, 'w') as f:
            f.writelines(adjusted_lines)
    
    def augment_minority_classes(self):
        """Augment minority classes to balance dataset"""
        class_counts = self.analyze_class_distribution()
        
        # Define augmentation strategies
        augmentations = ['brightness', 'contrast', 'blur', 'noise', 'hue_shift', 'horizontal_flip']
        
        print("\nStarting minority class augmentation...")
        
        for class_id in range(7):
            current_count = class_counts.get(class_id, 0)
            needed = max(0, self.target_samples - current_count)
            
            if needed == 0:
                continue
                
            class_name = self.class_names[class_id]
            print(f"\nAugmenting {class_name}: need {needed} more samples")
            
            # Get images containing this class
            images_with_class = self.get_images_with_class(class_id)
            
            if not images_with_class:
                print(f"  Warning: No images found for {class_name}")
                continue
            
            generated = 0
            attempts = 0
            max_attempts = needed * 3  # Safety limit
            
            train_images = self.data_root / "train" / "images"
            train_labels = self.data_root / "train" / "labels"
            
            while generated < needed and attempts < max_attempts:
                # Randomly select source image
                img_path, lbl_path = random.choice(images_with_class)
                
                # Randomly select augmentation
                aug_type = random.choice(augmentations)
                
                # Generate new filenames
                base_name = img_path.stem
                new_name = f"{base_name}_aug_{class_name}_{generated}_{aug_type}"
                new_img_path = train_images / f"{new_name}.jpg"
                new_lbl_path = train_labels / f"{new_name}.txt"
                
                # Apply augmentation
                try:
                    if self.augment_image(img_path, new_img_path, aug_type):
                        self.augment_labels(lbl_path, new_lbl_path, aug_type)
                        generated += 1
                        
                        if generated % 50 == 0:
                            print(f"  Generated {generated}/{needed} samples for {class_name}")
                
                except Exception as e:
                    print(f"  Error augmenting {img_path}: {e}")
                
                attempts += 1
            
            print(f"  Completed: {generated} new samples for {class_name}")

def create_balanced_subset(data_root, subset_size_per_class=200):
    """Create a smaller, balanced subset for faster experimentation"""
    data_root = Path(data_root)
    
    # Create balanced subset directory
    balanced_dir = data_root.parent / "aquarium_balanced"
    for split in ['train', 'valid']:
        (balanced_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (balanced_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Creating balanced subset in {balanced_dir}")
    
    # Copy data.yaml
    shutil.copy(data_root / "data.yaml", balanced_dir / "data.yaml")
    
    # Update paths in data.yaml
    with open(balanced_dir / "data.yaml", 'r') as f:
        content = f.read()
    
    content = content.replace(str(data_root), str(balanced_dir))
    
    with open(balanced_dir / "data.yaml", 'w') as f:
        f.write(content)
    
    # Process train and valid splits
    for split in ['train', 'valid']:
        print(f"\nProcessing {split} split...")
        
        # Track samples per class
        class_samples = {i: [] for i in range(7)}
        
        # Collect images by class
        labels_dir = data_root / split / 'labels'
        for label_file in labels_dir.glob("*.txt"):
            classes_in_image = set()
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)
            
            # Add image to each class it contains
            img_file = data_root / split / 'images' / f"{label_file.stem}.jpg"
            if img_file.exists():
                for class_id in classes_in_image:
                    if len(class_samples[class_id]) < subset_size_per_class:
                        class_samples[class_id].append((img_file, label_file))
        
        # Copy selected files
        copied_files = set()
        for class_id, samples in class_samples.items():
            class_name = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'][class_id]
            print(f"  {class_name}: {len(samples)} samples")
            
            for img_file, lbl_file in samples:
                if img_file.name not in copied_files:
                    # Copy image and label
                    shutil.copy(img_file, balanced_dir / split / 'images' / img_file.name)
                    shutil.copy(lbl_file, balanced_dir / split / 'labels' / lbl_file.name)
                    copied_files.add(img_file.name)
        
        print(f"  Total files copied: {len(copied_files)}")
    
    print(f"\nBalanced subset created in: {balanced_dir}")
    return balanced_dir

if __name__ == "__main__":
    data_root = "/Users/caoducanh/Desktop/Coding/UnderwaterObjectDetection/aquarium_pretrain"
    
    print("=== Data Preprocessing for Class Imbalance ===")
    print("\nChoose preprocessing approach:")
    print("1. Create balanced subset (recommended for experimentation)")
    print("2. Augment minority classes in full dataset")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n--- Creating Balanced Subset ---")
        balanced_dir = create_balanced_subset(data_root, subset_size_per_class=150)
        print(f"Balanced subset ready for training: {balanced_dir}")
    
    if choice in ['2', '3']:
        print("\n--- Augmenting Minority Classes ---")
        augmenter = MinorityClassAugmenter(data_root, target_samples_per_class=500)
        augmenter.augment_minority_classes()
        print("Minority class augmentation completed!")
    
    print("\n=== Preprocessing Complete ===")
    if choice in ['1', '3']:
        print(f"Use balanced dataset: {balanced_dir}/data.yaml")
    if choice in ['2', '3']:
        print(f"Use augmented dataset: {data_root}/data.yaml")
    
    print("\nNext steps:")
    print("1. Install required packages: pip install ultralytics torch torchvision")
    print("2. Run training: python train_yolov8.py")
    print("3. Monitor training plots for convergence")
    print("4. Evaluate per-class performance, especially minority classes")
