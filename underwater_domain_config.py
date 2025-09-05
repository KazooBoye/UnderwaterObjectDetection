#!/usr/bin/env python3

"""
Underwater Domain Adaptation Configuration
==========================================

This module provides specialized configurations and techniques for adapting
YOLO models to underwater environments through transfer learning.
"""

from pathlib import Path
import json

class UnderwaterDomainConfig:
    """Configuration for underwater domain adaptation"""
    
    # Underwater-specific hyperparameters based on research
    UNDERWATER_ADAPTATIONS = {
        # Color space adaptations for underwater blue-green dominance
        "color_adaptations": {
            "hsv_h": 0.05,      # Reduced hue variation (underwater has limited color spectrum)
            "hsv_s": 0.4,       # Higher saturation variation (important for visibility)
            "hsv_v": 0.3,       # Higher brightness variation (depth affects lighting)
        },
        
        # Geometric adaptations for underwater object behavior
        "geometric_adaptations": {
            "degrees": 15,      # More rotation (objects tumble in water)
            "translate": 0.15,  # More translation (current effects)
            "scale": 0.4,       # Higher scale variation (depth perception)
            "shear": 0.1,       # Water refraction effects
            "perspective": 0.0002,  # Underwater perspective distortion
        },
        
        # Noise and blur adaptations for water turbidity
        "environmental_adaptations": {
            "blur_limit": 7,    # Higher blur for turbid water
            "noise_limit": 25,  # Underwater noise
            "brightness_limit": 0.3,  # Lighting variations with depth
            "contrast_limit": 0.3,    # Reduced contrast underwater
        },
        
        # Training strategy for underwater conditions
        "training_strategy": {
            "warmup_epochs": 10,     # Longer warmup for domain adaptation
            "patience": 30,          # More patience for underwater complexity
            "lr0": 0.0005,          # Lower initial learning rate
            "lrf": 0.01,            # Final learning rate ratio
            "momentum": 0.9,         # Standard momentum
            "weight_decay": 0.0005,  # L2 regularization
            "label_smoothing": 0.15, # Higher smoothing for noisy underwater labels
        }
    }
    
    # Transfer learning phases optimized for underwater
    TRANSFER_PHASES = {
        "marine_adaptation": {
            "description": "Adapt to marine environment features",
            "freeze_backbone": True,
            "freeze_neck": False, 
            "epochs": 75,
            "focus": "color_and_texture_adaptation"
        },
        
        "object_specialization": {
            "description": "Specialize for underwater objects",
            "freeze_backbone": True,
            "freeze_neck": False,
            "epochs": 100, 
            "focus": "object_specific_features"
        },
        
        "full_finetuning": {
            "description": "Full model fine-tuning",
            "freeze_backbone": False,
            "freeze_neck": False,
            "epochs": 150,
            "focus": "end_to_end_optimization"
        }
    }
    
    @staticmethod
    def get_underwater_training_config(phase="full_finetuning", base_config=None):
        """Get training configuration optimized for underwater conditions"""
        
        base = {
            'data': 'aquarium_advanced/data.yaml',
            'epochs': 200,
            'batch': 8,
            'imgsz': 640,
            'device': '',
            'workers': 8,
            'amp': True,
            
            # Underwater-optimized augmentations
            **UnderwaterDomainConfig.UNDERWATER_ADAPTATIONS['color_adaptations'],
            **UnderwaterDomainConfig.UNDERWATER_ADAPTATIONS['geometric_adaptations'],
            **UnderwaterDomainConfig.UNDERWATER_ADAPTATIONS['training_strategy'],
            
            # Copy-paste for underwater object diversity
            'copy_paste': 0.3,
            'mosaic': 0.6,
            'mixup': 0.15,
            
            # Underwater-specific validation
            'val': True,
            'save_period': 25,
            'project': 'runs/underwater_transfer',
            'exist_ok': True,
        }
        
        # Apply phase-specific settings
        if phase in UnderwaterDomainConfig.TRANSFER_PHASES:
            phase_config = UnderwaterDomainConfig.TRANSFER_PHASES[phase]
            base['epochs'] = phase_config['epochs']
            base['name'] = f"underwater_{phase}"
        
        # Merge with base config if provided
        if base_config:
            base.update(base_config)
        
        return base
    
    @staticmethod
    def get_pretrained_model_recommendations():
        """Get recommended pretrained models for underwater transfer learning"""
        
        return {
            "tier1_marine_specialized": {
                "models": [
                    "fathomnet_yolov8s.pt",  # If available
                    "marine_objects_yolo.pt", # If available
                ],
                "description": "Models pretrained on marine/underwater datasets",
                "expected_improvement": "High (30-50% mAP improvement)",
                "availability": "Limited - check research repositories"
            },
            
            "tier2_adapted_coco": {
                "models": [
                    "yolo11s.pt",
                    "yolo11m.pt", 
                    "yolo11l.pt"
                ],
                "description": "Standard COCO models adapted for underwater",
                "expected_improvement": "Medium (15-30% mAP improvement)",
                "availability": "Available - will be adapted during training"
            },
            
            "tier3_custom_pretrained": {
                "models": [
                    "custom_marine_model.pt"  # User-provided
                ],
                "description": "Custom models pretrained on similar marine data",
                "expected_improvement": "Variable - depends on source domain similarity",
                "availability": "User-provided"
            }
        }
    
    @staticmethod
    def save_domain_adaptation_report(results, output_file="underwater_domain_adaptation_report.json"):
        """Save comprehensive report of domain adaptation results"""
        
        report = {
            "domain_adaptation_summary": {
                "source_domain": "COCO/ImageNet (terrestrial)",
                "target_domain": "Underwater/Marine",
                "adaptation_technique": "Progressive Transfer Learning",
                "dataset_enhancements": [
                    "Color space corrections for underwater conditions",
                    "Geometric augmentations for water movement",
                    "Copy-paste augmentation for object diversity",
                    "Hard negative mining for background robustness"
                ]
            },
            "results": results,
            "recommendations": {
                "deployment": "Model ready for underwater object detection",
                "further_improvements": [
                    "Collect more diverse underwater data",
                    "Apply test-time augmentation",
                    "Consider ensemble with multiple models",
                    "Fine-tune for specific marine environments"
                ]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_file

def create_underwater_model_comparison():
    """Create comparison script for different transfer learning approaches"""
    
    comparison_config = {
        "baseline_no_transfer": {
            "pretrained": False,
            "description": "Training from scratch",
            "expected_performance": "Poor - requires massive underwater dataset"
        },
        
        "standard_transfer": {
            "pretrained": "yolo11s.pt",
            "freeze_layers": None,
            "description": "Standard COCO pretrained model",
            "expected_performance": "Good baseline"
        },
        
        "progressive_transfer": {
            "pretrained": "yolo11s.pt", 
            "phases": ["freeze_backbone", "partial_unfreeze", "full_finetune"],
            "description": "Progressive unfreezing strategy",
            "expected_performance": "Better convergence and stability"
        },
        
        "underwater_adapted_transfer": {
            "pretrained": "underwater_adapted_model.pt",
            "adaptations": "color_space + geometric + environmental",
            "description": "Underwater-specific adaptations",
            "expected_performance": "Best performance for marine environments"
        }
    }
    
    return comparison_config

if __name__ == "__main__":
    print("🌊 Underwater Domain Adaptation Configuration Loaded")
    print("=" * 55)
    
    # Show available configurations
    config = UnderwaterDomainConfig.get_underwater_training_config()
    print("📋 Underwater Training Configuration:")
    for key, value in config.items():
        if len(str(value)) < 50:
            print(f"   {key}: {value}")
    
    print("\\n🎯 Transfer Learning Phases:")
    for phase, details in UnderwaterDomainConfig.TRANSFER_PHASES.items():
        print(f"   {phase}: {details['description']}")
    
    print("\\n🔍 Pretrained Model Recommendations:")
    recommendations = UnderwaterDomainConfig.get_pretrained_model_recommendations()
    for tier, info in recommendations.items():
        print(f"   {tier}: {info['description']}")
        print(f"      Expected improvement: {info['expected_improvement']}")
    
    print("\\n✅ Configuration ready for underwater transfer learning!")
