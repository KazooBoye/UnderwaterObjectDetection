"""
Advanced Data Preprocessing Pipeline for Underwater Object Detection

This script implements comprehensive data preprocessing techniques to improve 
underwater object detection performance by addressing dataset challenges:

1. Class imbalance through intelligent oversampling
2. Limited diversity through advanced augmentation
3. Poor visibility through image enhancement  
4. Background confusion through hard negative mining
5. Occlusion handling through synthetic techniques

Author: GitHub Copilot
Usage: python advanced_data_preprocessing.py
"""

import os
import cv2
import numpy as np
import random
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import albumentations as A
from albumentations import BboxParams
import math


class AdvancedUnderwaterPreprocessor:
    """Advanced preprocessing pipeline for underwater object detection"""
    
    def __init__(self, source_dir="aquarium_pretrain", output_dir="aquarium_advanced"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        self.stats = defaultdict(int)
        
        # Create advanced augmentation pipeline
        self.setup_augmentations()
        
        # Initialize background pool for copy-paste
        self.background_pool = []
        self.object_pool = defaultdict(list)  # Objects by class for copy-paste
        
        print(f"🌊 Advanced Underwater Preprocessor initialized")
        print(f"   Source: {source_dir}")
        print(f"   Output: {output_dir}")
    
    def setup_augmentations(self):
        """Setup comprehensive augmentation pipelines"""
        
        # 1. Geometric Augmentations (Strong)
        self.geometric_augment = A.Compose([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # Less common for underwater
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=10, 
                p=0.5
            ),
            A.RandomCrop(height=640, width=640, p=0.3),
            A.Resize(640, 640)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 2. Photometric Augmentations (Underwater-specific)  
        self.photometric_augment = A.Compose([
            # Color space manipulations
            A.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.3, 
                hue=0.1, 
                p=0.7
            ),
            # Underwater-specific enhancements
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=15, 
                p=0.5
            ),
            # Noise and blur for robustness
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=5, p=0.2),
            ], p=0.3),
            A.GaussNoise(p=0.2),
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 3. Light augmentations for copy-paste backgrounds
        self.light_augment = A.Compose([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.3),
        ])
        
        # 4. Underwater color correction pipeline
        self.color_correction = A.Compose([
            # Basic color adjustments
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        ])
    
    def analyze_dataset(self):
        """Analyze current dataset for preprocessing strategy"""
        print("🔍 Analyzing dataset characteristics...")
        
        class_distribution = Counter()
        object_sizes = defaultdict(list)
        image_qualities = []
        
        for split in ['train', 'valid']:
            labels_dir = self.source_dir / split / 'labels'
            images_dir = self.source_dir / split / 'images'
            
            if not labels_dir.exists():
                continue
            
            for label_file in tqdm(list(labels_dir.glob('*.txt')), desc=f"Analyzing {split}"):
                # Analyze annotations
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(float(parts[0]))
                                width = float(parts[3])
                                height = float(parts[4])
                                area = width * height
                                
                                class_distribution[class_id] += 1
                                object_sizes[class_id].append(area)
                except Exception:
                    continue
                
                # Analyze image quality (basic metrics)
                img_file = images_dir / f"{label_file.stem}.jpg"
                if img_file.exists():
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            brightness = np.mean(gray)
                            contrast = np.std(gray)
                            blue_dominance = np.mean(img[:,:,0]) / (np.mean(img) + 1e-8)
                            
                            image_qualities.append({
                                'brightness': brightness,
                                'contrast': contrast,
                                'blue_dominance': blue_dominance
                            })
                    except Exception:
                        continue
        
        # Calculate statistics
        total_objects = sum(class_distribution.values())
        max_class_count = max(class_distribution.values()) if class_distribution else 0
        min_class_count = min(class_distribution.values()) if class_distribution else 0
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        avg_brightness = np.mean([q['brightness'] for q in image_qualities])
        avg_contrast = np.mean([q['contrast'] for q in image_qualities])
        avg_blue_dominance = np.mean([q['blue_dominance'] for q in image_qualities])
        
        analysis = {
            'class_distribution': dict(class_distribution),
            'total_objects': total_objects,
            'imbalance_ratio': imbalance_ratio,
            'object_sizes': {k: np.mean(v) for k, v in object_sizes.items()},
            'image_quality': {
                'avg_brightness': avg_brightness,
                'avg_contrast': avg_contrast,
                'avg_blue_dominance': avg_blue_dominance
            }
        }
        
        print(f"\n📊 Dataset Analysis Results:")
        print(f"   Total objects: {total_objects}")
        print(f"   Class imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"   Average brightness: {avg_brightness:.1f}")
        print(f"   Average contrast: {avg_contrast:.1f}")
        print(f"   Blue dominance: {avg_blue_dominance:.2f}")
        
        print(f"\n📋 Class Distribution:")
        for class_id, count in sorted(class_distribution.items()):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            percentage = (count / total_objects) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return analysis
    
    def load_image_and_annotations(self, img_path, label_path):
        """Load image and corresponding annotations"""
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            return None, None, None
        
        # Load annotations
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(float(parts[0]))
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to albumentations format (x_center, y_center, width, height)
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            except Exception as e:
                print(f"Error loading annotations from {label_path}: {e}")
                return image, [], []
        
        return image, bboxes, class_labels
    
    def save_augmented_sample(self, image, bboxes, class_labels, output_img_path, output_label_path):
        """Save augmented image and annotations"""
        # Save image
        cv2.imwrite(str(output_img_path), image)
        
        # Save annotations
        with open(output_label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\\n")
        
        self.stats['images_generated'] += 1
    
    def extract_objects_for_copy_paste(self):
        """Extract objects and backgrounds for copy-paste augmentation"""
        print("🎭 Extracting objects and backgrounds for copy-paste augmentation...")
        
        train_images_dir = self.source_dir / 'train' / 'images'
        train_labels_dir = self.source_dir / 'train' / 'labels'
        
        if not train_images_dir.exists():
            return
        
        image_files = list(train_images_dir.glob('*.jpg'))
        
        for img_file in tqdm(image_files[:200], desc="Extracting objects"):  # Limit for performance
            image, bboxes, class_labels = self.load_image_and_annotations(
                img_file, 
                train_labels_dir / f"{img_file.stem}.txt"
            )
            
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Extract objects
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                
                # Convert to pixel coordinates
                x_center_px = int(x_center * w)
                y_center_px = int(y_center * h)
                width_px = int(width * w)
                height_px = int(height * h)
                
                # Calculate crop coordinates
                x1 = max(0, x_center_px - width_px // 2)
                y1 = max(0, y_center_px - height_px // 2)
                x2 = min(w, x_center_px + width_px // 2)
                y2 = min(h, y_center_px + height_px // 2)
                
                # Extract object
                if x2 > x1 and y2 > y1:
                    obj_crop = image[y1:y2, x1:x2]
                    if obj_crop.size > 0:
                        self.object_pool[class_id].append({
                            'image': obj_crop,
                            'original_size': (width, height),
                            'class_id': class_id
                        })
            
            # Store background (with objects masked out for cleaner backgrounds)
            if len(bboxes) > 0:
                background = image.copy()
                # Simple masking - this could be improved with inpainting
                for bbox in bboxes:
                    x_center, y_center, width, height = bbox
                    x_center_px = int(x_center * w)
                    y_center_px = int(y_center * h)
                    width_px = int(width * w)
                    height_px = int(height * h)
                    
                    x1 = max(0, x_center_px - width_px // 2)
                    y1 = max(0, y_center_px - height_px // 2)
                    x2 = min(w, x_center_px + width_px // 2)
                    y2 = min(h, y_center_px + height_px // 2)
                    
                    # Apply Gaussian blur to mask objects
                    roi = background[y1:y2, x1:x2]
                    if roi.size > 0:
                        blurred = cv2.GaussianBlur(roi, (15, 15), 0)
                        background[y1:y2, x1:x2] = blurred
                
                self.background_pool.append(background)
        
        print(f"   Extracted objects: {sum(len(v) for v in self.object_pool.values())}")
        print(f"   Background images: {len(self.background_pool)}")
        
        # Print objects per class
        for class_id, objects in self.object_pool.items():
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            print(f"     {class_name}: {len(objects)} objects")
    
    def create_copy_paste_augmentation(self, background_img, target_class_id, num_objects=3):
        """Create copy-paste augmented image"""
        if target_class_id not in self.object_pool or not self.object_pool[target_class_id]:
            return None, [], []
        
        augmented_img = background_img.copy()
        h, w = augmented_img.shape[:2]
        
        new_bboxes = []
        new_class_labels = []
        
        # Paste random objects
        for _ in range(random.randint(1, num_objects)):
            obj_data = random.choice(self.object_pool[target_class_id])
            obj_img = obj_data['image']
            obj_h, obj_w = obj_img.shape[:2]
            
            # Random scaling
            scale_factor = random.uniform(0.7, 1.5)
            new_obj_w = int(obj_w * scale_factor)
            new_obj_h = int(obj_h * scale_factor)
            
            if new_obj_w > 0 and new_obj_h > 0:
                obj_resized = cv2.resize(obj_img, (new_obj_w, new_obj_h))
                
                # Random position (avoid edges)
                margin = 50
                if w > new_obj_w + 2*margin and h > new_obj_h + 2*margin:
                    paste_x = random.randint(margin, w - new_obj_w - margin)
                    paste_y = random.randint(margin, h - new_obj_h - margin)
                    
                    # Paste with blending
                    alpha = random.uniform(0.8, 1.0)  # Slight transparency for realism
                    
                    roi = augmented_img[paste_y:paste_y+new_obj_h, paste_x:paste_x+new_obj_w]
                    if roi.shape[:2] == obj_resized.shape[:2]:
                        blended = cv2.addWeighted(roi, 1-alpha, obj_resized, alpha, 0)
                        augmented_img[paste_y:paste_y+new_obj_h, paste_x:paste_x+new_obj_w] = blended
                        
                        # Create bounding box in YOLO format
                        x_center = (paste_x + new_obj_w // 2) / w
                        y_center = (paste_y + new_obj_h // 2) / h
                        bbox_width = new_obj_w / w
                        bbox_height = new_obj_h / h
                        
                        new_bboxes.append([x_center, y_center, bbox_width, bbox_height])
                        new_class_labels.append(target_class_id)
        
        return augmented_img, new_bboxes, new_class_labels
    
    def generate_hard_negatives(self, num_negatives=500):
        """Generate hard negative samples (background-only images)"""
        print(f"🚫 Generating {num_negatives} hard negative samples...")
        
        hard_neg_dir = self.output_dir / 'train' / 'images'
        hard_neg_labels_dir = self.output_dir / 'train' / 'labels'
        
        for i in tqdm(range(num_negatives), desc="Generating hard negatives"):
            if self.background_pool:
                # Use existing backgrounds
                background = random.choice(self.background_pool)
                
                # Apply strong augmentations to create variety
                try:
                    augmented = self.photometric_augment(image=background)
                    final_img = augmented['image']
                except:
                    final_img = background
                
                # Save as hard negative
                img_name = f"hard_neg_{i:04d}.jpg"
                label_name = f"hard_neg_{i:04d}.txt"
                
                cv2.imwrite(str(hard_neg_dir / img_name), final_img)
                
                # Create empty label file
                with open(hard_neg_labels_dir / label_name, 'w') as f:
                    pass  # Empty file for hard negatives
                
                self.stats['hard_negatives'] += 1
    
    def apply_intelligent_oversampling(self, target_multiplier=3):
        """Apply intelligent oversampling for minority classes"""
        print("⚖️ Applying intelligent oversampling for class balance...")
        
        # Analyze class distribution
        class_distribution = Counter()
        
        train_labels_dir = self.source_dir / 'train' / 'labels'
        for label_file in train_labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(float(parts[0]))
                            class_distribution[class_id] += 1
            except:
                continue
        
        if not class_distribution:
            return
        
        max_count = max(class_distribution.values())
        minority_classes = []
        
        for class_id, count in class_distribution.items():
            if count < max_count // 2:  # Classes with less than half the max count
                minority_classes.append(class_id)
        
        print(f"   Minority classes to oversample: {[self.class_names[c] for c in minority_classes]}")
        
        # Oversample minority classes
        train_images_dir = self.source_dir / 'train' / 'images'
        image_files = list(train_images_dir.glob('*.jpg'))
        
        for img_file in tqdm(image_files, desc="Oversampling minorities"):
            image, bboxes, class_labels = self.load_image_and_annotations(
                img_file,
                train_labels_dir / f"{img_file.stem}.txt"
            )
            
            if image is None:
                continue
            
            # Check if image contains minority classes
            has_minority = any(cls in minority_classes for cls in class_labels)
            
            if has_minority:
                # Generate multiple augmented versions
                for aug_idx in range(target_multiplier):
                    try:
                        # Apply strong augmentations
                        if random.random() < 0.5:
                            augmented = self.geometric_augment(
                                image=image, 
                                bboxes=bboxes, 
                                class_labels=class_labels
                            )
                        else:
                            augmented = self.photometric_augment(
                                image=image, 
                                bboxes=bboxes, 
                                class_labels=class_labels
                            )
                        
                        if augmented['bboxes']:  # Only save if objects remain
                            output_name = f"{img_file.stem}_minority_aug_{aug_idx}"
                            self.save_augmented_sample(
                                augmented['image'],
                                augmented['bboxes'],
                                augmented['class_labels'],
                                self.output_dir / 'train' / 'images' / f"{output_name}.jpg",
                                self.output_dir / 'train' / 'labels' / f"{output_name}.txt"
                            )
                            
                            self.stats['minority_oversampled'] += 1
                    
                    except Exception as e:
                        continue
    
    def apply_copy_paste_augmentation(self, num_synthetic=1000):
        """Apply copy-paste augmentation focusing on starfish (minority class)"""
        print(f"🎭 Applying copy-paste augmentation ({num_synthetic} synthetic images)...")
        
        if not self.background_pool:
            print("   No backgrounds available for copy-paste")
            return
        
        # Focus on starfish (class 5) and other minority classes
        target_classes = [5, 1, 2, 3, 4, 6]  # All except fish (0)
        
        for i in tqdm(range(num_synthetic), desc="Copy-paste synthesis"):
            target_class = random.choice(target_classes)
            background = random.choice(self.background_pool)
            
            # Apply light augmentation to background
            try:
                bg_augmented = self.light_augment(image=background)
                background = bg_augmented['image']
            except:
                pass
            
            # Create copy-paste augmentation
            synthetic_img, new_bboxes, new_class_labels = self.create_copy_paste_augmentation(
                background, target_class, num_objects=random.randint(1, 4)
            )
            
            if synthetic_img is not None and new_bboxes:
                output_name = f"synthetic_copypaste_{i:04d}"
                self.save_augmented_sample(
                    synthetic_img,
                    new_bboxes,
                    new_class_labels,
                    self.output_dir / 'train' / 'images' / f"{output_name}.jpg",
                    self.output_dir / 'train' / 'labels' / f"{output_name}.txt"
                )
                
                self.stats['copy_paste_generated'] += 1
    
    def apply_advanced_augmentations(self, augmentation_factor=2):
        """Apply advanced augmentations to existing training data"""
        print(f"🎨 Applying advanced augmentations (factor: {augmentation_factor}x)...")
        
        train_images_dir = self.source_dir / 'train' / 'images'
        train_labels_dir = self.source_dir / 'train' / 'labels'
        
        if not train_images_dir.exists():
            return
        
        image_files = list(train_images_dir.glob('*.jpg'))
        
        for img_file in tqdm(image_files, desc="Advanced augmentation"):
            image, bboxes, class_labels = self.load_image_and_annotations(
                img_file,
                train_labels_dir / f"{img_file.stem}.txt"
            )
            
            if image is None:
                continue
            
            # Apply multiple augmentation strategies
            augmentation_types = [
                ('geometric', self.geometric_augment),
                ('photometric', self.photometric_augment),
                ('color_correction', self.color_correction),
            ]
            
            for aug_idx in range(augmentation_factor):
                aug_name, aug_pipeline = random.choice(augmentation_types)
                
                try:
                    augmented = aug_pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    if augmented['bboxes']:  # Only save if objects remain
                        output_name = f"{img_file.stem}_{aug_name}_{aug_idx}"
                        self.save_augmented_sample(
                            augmented['image'],
                            augmented['bboxes'],
                            augmented['class_labels'],
                            self.output_dir / 'train' / 'images' / f"{output_name}.jpg",
                            self.output_dir / 'train' / 'labels' / f"{output_name}.txt"
                        )
                        
                        self.stats[f'{aug_name}_augmented'] += 1
                
                except Exception as e:
                    continue
    
    def copy_original_data(self):
        """Copy original data to output directory"""
        print("📁 Copying original dataset...")
        
        # Create output directory structure
        for split in ['train', 'valid', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy all original data
        for split in ['train', 'valid', 'test']:
            source_images = self.source_dir / split / 'images'
            source_labels = self.source_dir / split / 'labels'
            dest_images = self.output_dir / split / 'images'
            dest_labels = self.output_dir / split / 'labels'
            
            if source_images.exists():
                for img_file in source_images.glob('*.jpg'):
                    shutil.copy2(img_file, dest_images / img_file.name)
                    self.stats['original_images_copied'] += 1
            
            if source_labels.exists():
                for label_file in source_labels.glob('*.txt'):
                    shutil.copy2(label_file, dest_labels / label_file.name)
                    self.stats['original_labels_copied'] += 1
    
    def create_data_yaml(self):
        """Create data.yaml for the enhanced dataset"""
        data_yaml_content = f"""# Advanced Preprocessed Underwater Dataset
train: train/images
val: valid/images
test: test/images

nc: {len(self.class_names)}
names: {self.class_names}

# Preprocessing applied:
# - Intelligent oversampling for class balance
# - Copy-paste augmentation for minority classes  
# - Advanced geometric and photometric augmentations
# - Hard negative mining
# - Underwater-specific color corrections
"""
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            f.write(data_yaml_content)
    
    def generate_preprocessing_report(self):
        """Generate comprehensive preprocessing report"""
        print("\\n📄 Generating preprocessing report...")
        
        report = {
            'preprocessing_stats': dict(self.stats),
            'techniques_applied': [
                'Intelligent oversampling for minority classes',
                'Copy-paste augmentation with extracted objects',
                'Advanced geometric transformations',
                'Underwater-specific photometric enhancements', 
                'Hard negative mining',
                'Color space corrections',
                'Multi-scale augmentations'
            ],
            'recommendations': [
                'Use longer training schedule (200+ epochs)',
                'Apply label smoothing (0.1) for overconfident predictions',
                'Consider focal loss for class imbalance',
                'Use test-time augmentation during inference',
                'Monitor validation carefully for overfitting'
            ]
        }
        
        with open(self.output_dir / 'preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Human-readable report
        with open(self.output_dir / 'preprocessing_report.txt', 'w') as f:
            f.write("🌊 ADVANCED UNDERWATER DATASET PREPROCESSING REPORT\\n")
            f.write("="*60 + "\\n\\n")
            
            f.write("📊 PREPROCESSING STATISTICS\\n")
            f.write("-" * 30 + "\\n")
            for key, value in self.stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\\n")
            
            f.write("\\n🎯 TECHNIQUES APPLIED\\n")
            f.write("-" * 30 + "\\n")
            for technique in report['techniques_applied']:
                f.write(f"• {technique}\\n")
            
            f.write("\\n💡 TRAINING RECOMMENDATIONS\\n")
            f.write("-" * 30 + "\\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\\n")
            
            f.write("\\n🚀 NEXT STEPS\\n")
            f.write("-" * 30 + "\\n")
            f.write("1. Review augmented samples visually\\n")
            f.write("2. Update training config with longer schedule\\n")
            f.write("3. Use focal loss or class weights for imbalance\\n")
            f.write("4. Monitor training with enhanced validation\\n")
            f.write("5. Apply test-time augmentation for inference\\n")
        
        print(f"✅ Preprocessing report saved to {self.output_dir}/preprocessing_report.txt")
    
    def run_complete_preprocessing(self):
        """Run the complete advanced preprocessing pipeline"""
        print("🚀 Starting Advanced Underwater Dataset Preprocessing...")
        print("="*60)
        
        # Step 1: Analyze original dataset
        analysis = self.analyze_dataset()
        
        # Step 2: Copy original data
        self.copy_original_data()
        
        # Step 3: Extract objects and backgrounds for copy-paste
        self.extract_objects_for_copy_paste()
        
        # Step 4: Apply intelligent oversampling
        self.apply_intelligent_oversampling(target_multiplier=2)
        
        # Step 5: Apply copy-paste augmentation 
        self.apply_copy_paste_augmentation(num_synthetic=800)
        
        # Step 6: Apply advanced augmentations
        self.apply_advanced_augmentations(augmentation_factor=1)
        
        # Step 7: Generate hard negatives
        self.generate_hard_negatives(num_negatives=300)
        
        # Step 8: Create data.yaml
        self.create_data_yaml()
        
        # Step 9: Generate report
        self.generate_preprocessing_report()
        
        print("\\n✅ ADVANCED PREPROCESSING COMPLETE!")
        print("="*60)
        print(f"📁 Enhanced dataset saved to: {self.output_dir}")
        print(f"📄 Report saved to: {self.output_dir}/preprocessing_report.txt")
        
        print("\\n📊 FINAL STATISTICS:")
        for key, value in sorted(self.stats.items()):
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        return self.stats


def create_optimized_training_config():
    """Create optimized training configuration for enhanced dataset"""
    
    config_content = '''
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
'''
    
    with open('advanced_config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Advanced training config saved to advanced_config.py")


def main():
    """Main execution function"""
    print("🌊 Advanced Underwater Dataset Preprocessing Pipeline")
    print("="*60)
    
    # Check if source dataset exists
    if not Path("aquarium_pretrain").exists():
        print("❌ Source dataset 'aquarium_pretrain' not found!")
        print("Please ensure your dataset is available before running preprocessing.")
        return
    
    try:
        # Initialize preprocessor
        preprocessor = AdvancedUnderwaterPreprocessor(
            source_dir="aquarium_pretrain",
            output_dir="aquarium_advanced"
        )
        
        # Run complete preprocessing pipeline
        stats = preprocessor.run_complete_preprocessing()
        
        # Create optimized training configuration
        create_optimized_training_config()
        
        print("\\n🎯 WHAT'S BEEN ACCOMPLISHED:")
        print("✅ Intelligent class balancing through oversampling")
        print("✅ Copy-paste augmentation for minority classes")
        print("✅ Advanced geometric and photometric augmentations")
        print("✅ Underwater-specific color corrections")
        print("✅ Hard negative mining for better background handling")
        print("✅ Multi-scale and rotation invariance")
        print("✅ Optimized training configuration")
        
        print("\\n🚀 NEXT STEPS:")
        print("1. Review sample augmented images in aquarium_advanced/")
        print("2. Start training with optimized config:")
        print("   python train_yolo_configured.py --config advanced_config")
        print("3. Monitor training with longer patience settings")
        print("4. Consider ensemble methods for final deployment")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        print("Please check the error details and ensure all dependencies are installed.")


if __name__ == "__main__":
    main()
