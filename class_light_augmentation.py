#!/usr/bin/env python3
"""
Simple Dataset Balancer - Light Augmentation Only
Alternative to heavy pre-augmentation approach
"""

import os
import shutil
import glob
import random
import cv2
import numpy as np
import albumentations as A
from collections import Counter, defaultdict
from pathlib import Path
import yaml

class SimpleDatasetBalancer:
    def __init__(self, source_dir='./aquarium_pretrain', target_dir='./aquarium_light_aug'):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        
        # Simple augmentation - minimal transforms
        self.light_augmentation = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.GaussNoise(p=0.1),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def load_image_and_annotations(self, image_path, label_path):
        """Load image and YOLO format annotations"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        return image, bboxes, class_labels
    
    def save_augmented_data(self, image, bboxes, class_labels, output_image_path, output_label_path):
        """Save augmented image and annotations"""
        # Save image
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image_bgr)
        
        # Save annotations
        with open(output_label_path, 'w') as f:
            for bbox, class_label in zip(bboxes, class_labels):
                f.write(f"{class_label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\\n")
    
    def analyze_dataset(self):
        """Quick dataset analysis"""
        print("📊 ANALYZING DATASET...")
        
        class_counts = Counter()
        for split in ['train', 'valid', 'test']:
            labels_dir = os.path.join(self.source_dir, split, 'labels')
            if os.path.exists(labels_dir):
                label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
        
        print("\\nClass Distribution:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  {self.class_names[class_id]}: {count}")
        
        return class_counts
    
    def create_balanced_dataset(self):
        """Create balanced dataset with minimal augmentation"""
        
        print("🚀 CREATING SIMPLY BALANCED DATASET")
        print("=" * 50)
        
        # Analyze original dataset
        class_counts = self.analyze_dataset()
        
        # Calculate target - use 75th percentile instead of heavy oversampling
        target_per_class = int(np.percentile(list(class_counts.values()), 75))
        print(f"\\n🎯 Target samples per class: {target_per_class}")
        
        # Remove existing target directory
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        
        # Create directory structure
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(self.target_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.target_dir, split, 'labels'), exist_ok=True)
        
        balanced_stats = Counter()
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            print(f"\\n📁 Processing {split.upper()} split...")
            
            source_images_dir = os.path.join(self.source_dir, split, 'images')
            source_labels_dir = os.path.join(self.source_dir, split, 'labels')
            target_images_dir = os.path.join(self.target_dir, split, 'images')
            target_labels_dir = os.path.join(self.target_dir, split, 'labels')
            
            if not os.path.exists(source_images_dir):
                print(f"⚠️ Skipping {split} - directory not found")
                continue
            
            # Get all image files
            image_files = glob.glob(os.path.join(source_images_dir, '*.jpg'))
            
            # Group images by classes they contain
            class_to_images = defaultdict(list)
            for image_file in image_files:
                image_name = os.path.basename(image_file)
                label_file = os.path.join(source_labels_dir, image_name.replace('.jpg', '.txt'))
                
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        classes_in_image = set()
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                classes_in_image.add(class_id)
                    
                    for class_id in classes_in_image:
                        class_to_images[class_id].append(image_file)
            
            # Copy original images
            copied_images = set()
            for image_file in image_files:
                image_name = os.path.basename(image_file)
                label_file = os.path.join(source_labels_dir, image_name.replace('.jpg', '.txt'))
                
                # Copy original
                shutil.copy2(image_file, target_images_dir)
                if os.path.exists(label_file):
                    shutil.copy2(label_file, target_labels_dir)
                    copied_images.add(image_file)
            
            # Light augmentation for minority classes only in training
            if split == 'train':
                for class_id, images in class_to_images.items():
                    class_name = self.class_names[class_id]
                    current_count = len([img for img in images if img in copied_images])
                    
                    # Only augment if significantly below target
                    if current_count < target_per_class * 0.7:
                        needed = min(target_per_class - current_count, current_count)  # Don't over-augment
                        
                        print(f"  🔧 Augmenting {class_name}: +{needed} samples")
                        
                        for i in range(needed):
                            source_image = random.choice(images)
                            source_name = os.path.basename(source_image)
                            source_label = os.path.join(source_labels_dir, source_name.replace('.jpg', '.txt'))
                            
                            if os.path.exists(source_label):
                                # Load and augment
                                image, bboxes, class_labels = self.load_image_and_annotations(source_image, source_label)
                                
                                try:
                                    augmented = self.light_augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
                                    
                                    # Save augmented version
                                    aug_name = f"{source_name.split('.')[0]}_aug_{i}.jpg"
                                    aug_image_path = os.path.join(target_images_dir, aug_name)
                                    aug_label_path = os.path.join(target_labels_dir, aug_name.replace('.jpg', '.txt'))
                                    
                                    self.save_augmented_data(
                                        augmented['image'],
                                        augmented['bboxes'],
                                        augmented['class_labels'],
                                        aug_image_path,
                                        aug_label_path
                                    )
                                    
                                except Exception as e:
                                    print(f"    ⚠️ Augmentation failed for {source_name}: {e}")
        
        # Create data.yaml
        self.create_data_yaml()
        
        # Final verification
        self.verify_dataset()
        
        return balanced_stats
    
    def create_data_yaml(self):
        """Create data.yaml configuration file"""
        data_yaml = {
            'path': os.path.abspath(self.target_dir),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = os.path.join(self.target_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✅ Created data.yaml: {yaml_path}")
    
    def verify_dataset(self):
        """Verify the balanced dataset"""
        print(f"\\n{'='*50}")
        print("📊 DATASET VERIFICATION")
        print("="*50)
        
        for split in ['train', 'valid', 'test']:
            labels_dir = os.path.join(self.target_dir, split, 'labels')
            if os.path.exists(labels_dir):
                class_counts = Counter()
                label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
                
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                
                print(f"\\n--- {split.upper()} ---")
                print(f"Images: {len(label_files)}")
                print(f"Total objects: {sum(class_counts.values())}")
                for class_id in sorted(class_counts.keys()):
                    count = class_counts[class_id]
                    print(f"  {self.class_names[class_id]}: {count}")
        
        print(f"\\n✅ SIMPLE BALANCED DATASET READY!")
        print(f"📁 Location: {os.path.abspath(self.target_dir)}")

def main():
    print("🎯 SIMPLE DATASET BALANCER")
    print("Light augmentation approach - avoids double augmentation")
    print("Suitable for letting YOLO handle most augmentation during training")
    
    balancer = SimpleDatasetBalancer()
    balancer.create_balanced_dataset()
    
    print(f"\\n✅ SIMPLE BALANCING COMPLETE!")
    print("Ready for normal YOLOv8 training with standard augmentation")

if __name__ == "__main__":
    main()
