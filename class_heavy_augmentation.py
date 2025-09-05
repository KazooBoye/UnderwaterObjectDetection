#!/usr/bin/env python3
"""
Advanced Dataset Balancer with Smart Augmentation
Addresses critical issues from dataset analysis:
- Severe class imbalance (25:1 ratio)
- Multi-scale object detection complexity
- Distribution drift across splits
- Limit        # Calculate target counts - use a more balanced approach
        all_counts = []
        for split_stats in dataset_stats.values():
            all_counts.extend(split_stats['objects'].values())
        
        # Use 70th percentile instead of median for better balance
        target_objects_per_class = int(np.percentile([count for count in all_counts if count > 0], 70))
        print(f"\n Target objects per class: {target_objects_per_class}")ning data for rare classes
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
import matplotlib.pyplot as plt

class AdvancedDatasetBalancer:
    def __init__(self, source_dir='./aquarium_pretrain', target_dir='./aquarium_heavy_aug'):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        
        # Class-specific augmentation strategies based on analysis
        self.class_strategies = {
            'fish': {'weight': 1.0, 'aug_factor': 1.0},  # Majority class - minimal augmentation
            'jellyfish': {'weight': 1.5, 'aug_factor': 2.0},  # Medium augmentation
            'penguin': {'weight': 2.0, 'aug_factor': 3.0},  # Higher augmentation
            'puffin': {'weight': 3.0, 'aug_factor': 4.0},  # Heavy augmentation
            'shark': {'weight': 2.5, 'aug_factor': 3.5},  # Heavy augmentation
            'starfish': {'weight': 8.0, 'aug_factor': 10.0},  # Maximum augmentation (25:1 ratio)
            'stingray': {'weight': 5.0, 'aug_factor': 6.0}  # Very heavy augmentation
        }
        
        # Size-aware augmentation parameters from analysis
        self.size_categories = {
            'small': {'threshold': 0.02, 'classes': ['fish', 'jellyfish', 'penguin']},
            'medium': {'threshold': 0.04, 'classes': ['puffin']},
            'large': {'threshold': 0.08, 'classes': ['shark', 'starfish', 'stingray']}
        }
    
    def create_class_specific_augmentations(self, class_name, object_size='medium'):
        """Create augmentation pipeline specific to class characteristics"""
        
        # Base augmentations for all classes
        base_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
        ]
        
        # Class-specific augmentations based on analysis findings
        specific_transforms = []  # Initialize to avoid unbound variable
        
        if class_name == 'fish':
            # Fish are often in groups/schools - minimal geometric transforms
            specific_transforms = [
                A.GaussNoise(p=0.2),  # Reduced noise
                A.Blur(blur_limit=3, p=0.2),
                A.RandomRotate90(p=0.3),  # Limited rotation for schooling fish
            ]
            
        elif class_name == 'jellyfish':
            # Translucent, need to preserve shape integrity
            specific_transforms = [
                A.GaussNoise(p=0.3),  # Reduced noise for translucent objects
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.RandomRotate90(p=0.5),
                A.Affine(shear=(-10, 10), rotate=(-15, 15), p=0.6),
            ]
            
        elif class_name in ['penguin', 'puffin']:
            # Birds - can handle more geometric transforms, minimal noise needed
            specific_transforms = [
                A.Affine(shear=(-15, 15), rotate=(-25, 25), translate_percent=(-0.15, 0.15), p=0.7),
                A.RandomRotate90(p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.OpticalDistortion(distort_limit=0.1, p=0.3),
                A.GaussNoise(p=0.2),  # Added minimal noise
            ]
            
        elif class_name == 'shark':
            # Large, elongated (2.12 aspect ratio) - preserve aspect ratio
            specific_transforms = [
                A.Affine(shear=(-10, 10), rotate=(-20, 20), translate_percent=(-0.1, 0.1), p=0.6),
                A.ElasticTransform(alpha=1, sigma=50, p=0.4),
                A.GaussNoise(p=0.25),  # Reduced noise
            ]
            
        elif class_name == 'starfish':
            # Most critical class (78 samples) - maximum safe augmentation
            specific_transforms = [
                A.Affine(shear=(-20, 20), rotate=(-30, 30), translate_percent=(-0.2, 0.2), p=0.8),
                A.RandomRotate90(p=0.7),
                A.Perspective(scale=(0.05, 0.15), p=0.5),
                A.ElasticTransform(alpha=2, sigma=50, p=0.5),
                A.OpticalDistortion(distort_limit=0.15, p=0.4),
                A.GaussNoise(p=0.3),  # Reduced noise even for critical class
            ]
            
        elif class_name == 'stingray':
            # Large, flat (2.06 aspect ratio) - preserve distinctive shape
            specific_transforms = [
                A.Affine(shear=(-15, 15), rotate=(-25, 25), translate_percent=(-0.15, 0.15), p=0.7),
                A.ElasticTransform(alpha=1.5, sigma=50, p=0.4),
                A.Perspective(scale=(0.05, 0.12), p=0.4),
                A.GaussNoise(p=0.25),  # Reduced noise
            ]
        
        # Size-specific adjustments
        if object_size == 'small':
            # For small objects, reduce aggressive transforms to preserve details
            size_transforms = [
                A.Affine(scale=(0.9, 1.1), p=0.3),  # Gentle scaling
                A.Blur(blur_limit=3, p=0.1),  # Minimal blur
            ]
        elif object_size == 'large':
            # Large objects can handle more aggressive transforms
            size_transforms = [
                A.Affine(scale=(0.8, 1.2), p=0.4),
                A.CoarseDropout(p=0.2),  # Simplified parameters
            ]
        else:
            size_transforms = [A.Affine(scale=(0.85, 1.15), p=0.3)]
        
        # Underwater-specific augmentations
        underwater_transforms = [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # Underwater lighting
            A.RandomToneCurve(scale=0.1, p=0.2),  # Alternative to RandomShadow
        ]
        
        return A.Compose(
            base_transforms + specific_transforms + size_transforms + underwater_transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
    
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
                x_center, y_center, width, height = bbox
                f.write(f"{int(class_label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def analyze_source_dataset(self):
        """Analyze source dataset to understand class distribution and characteristics"""
        print(" ANALYZING SOURCE DATASET...")
        
        dataset_stats = {}
        
        for split in ['train', 'valid', 'test']:
            labels_dir = os.path.join(self.source_dir, split, 'labels')
            if not os.path.exists(labels_dir):
                continue
                
            split_stats = {
                'images': 0,
                'objects': Counter(),
                'object_sizes': defaultdict(list),
                'class_images': defaultdict(set)  # Track which images contain each class
            }
            
            label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
            split_stats['images'] = len(label_files)
            
            for label_file in label_files:
                image_name = os.path.splitext(os.path.basename(label_file))[0]
                classes_in_image = set()
                
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            width, height = float(parts[3]), float(parts[4])
                            area = width * height
                            
                            split_stats['objects'][class_id] += 1
                            split_stats['object_sizes'][class_id].append(area)
                            classes_in_image.add(class_id)
                
                # Track which images contain each class
                for class_id in classes_in_image:
                    split_stats['class_images'][class_id].add(image_name)
            
            dataset_stats[split] = split_stats
        
        # Print analysis
        print(f"\n DATASET ANALYSIS RESULTS:")
        for split, stats in dataset_stats.items():
            print(f"\n--- {split.upper()} SPLIT ---")
            print(f"Images: {stats['images']}")
            print(f"Total objects: {sum(stats['objects'].values())}")
            
            for class_id in range(len(self.class_names)):
                count = stats['objects'].get(class_id, 0)
                images_with_class = len(stats['class_images'].get(class_id, set()))
                avg_size = np.mean(stats['object_sizes'].get(class_id, [0])) if class_id in stats['object_sizes'] else 0
                
                print(f"  {self.class_names[class_id]:<12}: {count:>4} objects in {images_with_class:>3} images (avg size: {avg_size:.4f})")
        
        return dataset_stats
    
    def create_balanced_dataset_with_augmentation(self):
        """Create balanced dataset using intelligent augmentation strategies"""
        
        print(" CREATING ADVANCED BALANCED DATASET")
        print("=" * 60)
        
        # Analyze source dataset
        dataset_stats = self.analyze_source_dataset()
        
        # Remove existing target directory
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        
        # Calculate target counts (use median of current distribution)
        all_counts = []
        for split_stats in dataset_stats.values():
            all_counts.extend(split_stats['objects'].values())
        
        target_objects_per_class = int(np.median([count for count in all_counts if count > 0]) * 2)
        print(f"\n Target objects per class: {target_objects_per_class}")
        
        balanced_stats = Counter()
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            if split not in dataset_stats:
                continue
                
            print(f"\n Processing {split} split...")
            
            source_images_dir = os.path.join(self.source_dir, split, 'images')
            source_labels_dir = os.path.join(self.source_dir, split, 'labels')
            target_images_dir = os.path.join(self.target_dir, split, 'images')
            target_labels_dir = os.path.join(self.target_dir, split, 'labels')
            
            os.makedirs(target_images_dir, exist_ok=True)
            os.makedirs(target_labels_dir, exist_ok=True)
            
            split_stats = dataset_stats[split]
            
            # Special handling for test split - just copy without any processing
            if split == 'test':
                print(f" Copying test split as-is (no processing needed)")
                
                # Copy all images
                if os.path.exists(source_images_dir):
                    for image_file in glob.glob(os.path.join(source_images_dir, '*')):
                        shutil.copy2(image_file, target_images_dir)
                
                # Copy all labels  
                if os.path.exists(source_labels_dir):
                    for label_file in glob.glob(os.path.join(source_labels_dir, '*')):
                        shutil.copy2(label_file, target_labels_dir)
                
                print(f" Copied {split_stats['images']} images")
                continue
            
            # Process train/valid splits with augmentation
            
            # Process each class
            for class_id, class_name in enumerate(self.class_names):
                current_count = split_stats['objects'].get(class_id, 0)
                if current_count == 0:
                    continue
                
                # Get images containing this class
                class_images = list(split_stats['class_images'].get(class_id, set()))
                
                if not class_images:
                    continue
                
                # Calculate how many augmentations needed
                split_target = target_objects_per_class if split == 'train' else max(10, target_objects_per_class // 10)
                
                if current_count >= split_target:
                    # Downsample: use subset of images
                    augmentation_factor = 1
                    target_images_needed = min(len(class_images), max(1, split_target // (current_count // len(class_images))))
                else:
                    # Upsample: generate more augmentations
                    augmentation_factor = max(1, int(split_target / current_count))
                    target_images_needed = len(class_images)
                
                print(f"  {class_name}: {current_count} → target {split_target} (x{augmentation_factor} augmentation)")
                
                # Create augmentation pipeline for this class
                avg_size = np.mean(split_stats['object_sizes'].get(class_id, [0.02]))
                size_category = 'small' if avg_size < 0.02 else 'large' if avg_size > 0.04 else 'medium'
                
                augment_pipeline = self.create_class_specific_augmentations(class_name, size_category)
                
                # Generate augmented samples
                samples_generated = 0
                attempts = 0
                max_attempts = len(class_images) * augmentation_factor * 3  # Safety limit
                
                while samples_generated < augmentation_factor * len(class_images) and attempts < max_attempts:
                    attempts += 1
                    
                    # Select random image containing this class
                    image_name = random.choice(class_images)
                    image_path = os.path.join(source_images_dir, f"{image_name}.jpg")
                    label_path = os.path.join(source_labels_dir, f"{image_name}.txt")
                    
                    if not os.path.exists(image_path):
                        continue
                    
                    try:
                        # Load image and annotations
                        image, bboxes, class_labels = self.load_image_and_annotations(image_path, label_path)
                        
                        if len(bboxes) == 0:
                            continue
                        
                        # Apply augmentation
                        augmented = augment_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                        
                        if len(augmented['bboxes']) == 0:
                            continue  # Skip if all bboxes were lost during augmentation
                        
                        # Save augmented data
                        aug_suffix = f"_{class_name}_aug_{samples_generated:04d}"
                        output_image_name = f"{image_name}{aug_suffix}.jpg"
                        output_label_name = f"{image_name}{aug_suffix}.txt"
                        
                        output_image_path = os.path.join(target_images_dir, output_image_name)
                        output_label_path = os.path.join(target_labels_dir, output_label_name)
                        
                        self.save_augmented_data(
                            augmented['image'], 
                            augmented['bboxes'], 
                            augmented['class_labels'],
                            output_image_path, 
                            output_label_path
                        )
                        
                        # Update statistics
                        for label in augmented['class_labels']:
                            balanced_stats[label] += 1
                        
                        samples_generated += 1
                        
                    except Exception as e:
                        print(f"    Warning: Failed to augment {image_name}: {str(e)}")
                        continue
                
                print(f"    Generated {samples_generated} augmented samples")
        
        # Create data.yaml
        self.create_data_yaml()
        
        # Final verification
        self.verify_balanced_dataset()
        
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
        
        print(f" Created data.yaml: {yaml_path}")
    
    def verify_balanced_dataset(self):
        """Verify the balanced dataset"""
        print(f"\n{'='*60}")
        print("ADVANCED BALANCED DATASET VERIFICATION")
        print("="*60)
        
        verification_stats = {}
        total_stats = Counter()
        
        for split in ['train', 'valid']:
            split_dir = os.path.join(self.target_dir, split, 'labels')
            if not os.path.exists(split_dir):
                continue
                
            split_stats = Counter()
            label_files = glob.glob(os.path.join(split_dir, '*.txt'))
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            split_stats[class_id] += 1
            
            verification_stats[split] = split_stats
            total_stats.update(split_stats)
            
            print(f"\n {split.upper()} split: {len(label_files)} images, {sum(split_stats.values())} objects")
            for class_id in range(len(self.class_names)):
                count = split_stats.get(class_id, 0)
                percentage = (count / sum(split_stats.values())) * 100 if sum(split_stats.values()) > 0 else 0
                print(f"  {self.class_names[class_id]:<12}: {count:>4} ({percentage:>5.1f}%)")
        
        # Overall balance analysis
        if total_stats:
            max_count = max(total_stats.values())
            min_count = min(total_stats.values())
            balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"\n OVERALL BALANCE ANALYSIS:")
            print(f"Total objects: {sum(total_stats.values())}")
            
            for class_id in range(len(self.class_names)):
                count = total_stats.get(class_id, 0)
                percentage = (count / sum(total_stats.values())) * 100
                print(f"  {self.class_names[class_id]:<12}: {count:>4} ({percentage:>5.1f}%)")
            
            print(f"\n Balance Ratio: {balance_ratio:.2f}:1")
            
            if balance_ratio < 2.0:
                print(" EXCELLENT balance achieved!")
            elif balance_ratio < 3.0:
                print(" GOOD balance achieved!")
            elif balance_ratio < 5.0:
                print(" MODERATE balance - better than original")
            else:
                print(" Some improvement, but still needs work")
            
            print(f"\n IMPROVEMENT ANALYSIS:")
            print(f"Original dataset: ~25:1 ratio (severe imbalance)")
            print(f"Balanced dataset: {balance_ratio:.2f}:1 ratio")
            improvement = 25 / balance_ratio if balance_ratio > 0 else float('inf')
            print(f"Improvement factor: {improvement:.1f}x better balance")
            
        print(f"\n DATASET READY FOR TRAINING!")
        print(f"Location: {os.path.abspath(self.target_dir)}")
        print(f"Configuration: {os.path.join(self.target_dir, 'data.yaml')}")

def main():
    print(" ADVANCED UNDERWATER DATASET BALANCER")
    print("Addresses critical issues from comprehensive dataset analysis")
    print("- Severe class imbalance (25:1 → <3:1)")
    print("- Multi-scale object detection complexity")
    print("- Class-specific augmentation strategies")
    print("- Underwater environment considerations")
    
    balancer = AdvancedDatasetBalancer()
    
    # Create balanced dataset with intelligent augmentation
    balanced_stats = balancer.create_balanced_dataset_with_augmentation()
    
    print(f"\n ADVANCED BALANCING COMPLETE!")
    print("Dataset is ready for improved YOLOv8 training with enhanced balance and preprocessing.")

if __name__ == "__main__":
    main()
