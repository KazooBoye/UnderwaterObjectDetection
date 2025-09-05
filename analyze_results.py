#!/usr/bin/env python3
"""
Training Results Analysis and Comparison Tool
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def analyze_training_results(results_dir, experiment_name):
    """Analyze training results from CSV file"""
    
    results_file = Path(results_dir) / 'results.csv'
    if not results_file.exists():
        print(f"❌ No results.csv found in {results_dir}")
        return None
    
    # Load results
    df = pd.read_csv(results_file)
    
    print(f"\n📊 ANALYSIS: {experiment_name}")
    print("=" * 60)
    
    # Basic stats
    total_epochs = len(df)
    best_epoch = df['metrics/mAP50(B)'].idxmax()
    best_map50 = df['metrics/mAP50(B)'].iloc[best_epoch]
    best_map50_95 = df['metrics/mAP50-95(B)'].iloc[best_epoch]
    final_map50 = df['metrics/mAP50(B)'].iloc[-1]
    final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1]
    
    print(f"📈 Training Epochs: {total_epochs}")
    print(f"🎯 Best mAP50: {best_map50:.3f} (epoch {best_epoch + 1})")
    print(f"🎯 Best mAP50-95: {best_map50_95:.3f} (epoch {best_epoch + 1})")
    print(f"📊 Final mAP50: {final_map50:.3f}")
    print(f"📊 Final mAP50-95: {final_map50_95:.3f}")
    
    # Check for overfitting
    val_box_loss_trend = np.polyfit(range(len(df)), df['val/box_loss'], 1)[0]
    train_box_loss_trend = np.polyfit(range(len(df)), df['train/box_loss'], 1)[0]
    
    if val_box_loss_trend > 0 and train_box_loss_trend < 0:
        print("⚠️ OVERFITTING DETECTED: Validation loss increasing while training loss decreasing")
    elif best_epoch < total_epochs * 0.7:
        print("⚠️ EARLY PEAK: Best performance in early epochs suggests overfitting")
    else:
        print("✅ Training appears stable")
    
    # Performance assessment
    if final_map50 < 0.5:
        print("🔴 POOR PERFORMANCE: mAP50 < 50%")
    elif final_map50 < 0.7:
        print("🟡 MODERATE PERFORMANCE: mAP50 50-70%")
    else:
        print("🟢 GOOD PERFORMANCE: mAP50 > 70%")
    
    return {
        'df': df,
        'best_epoch': best_epoch,
        'best_map50': best_map50,
        'best_map50_95': best_map50_95,
        'final_map50': final_map50,
        'final_map50_95': final_map50_95,
        'overfitting': val_box_loss_trend > 0 and train_box_loss_trend < 0
    }

def plot_training_comparison(results_list, experiment_names):
    """Plot comparison of multiple training runs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (results, name) in enumerate(zip(results_list, experiment_names)):
        if results is None:
            continue
            
        df = results['df']
        color = colors[i % len(colors)]
        
        # mAP50 progression
        axes[0, 0].plot(df.index + 1, df['metrics/mAP50(B)'], 
                       label=name, color=color, linewidth=2)
        axes[0, 0].set_title('mAP50 Progression')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('mAP50')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss comparison
        axes[0, 1].plot(df.index + 1, df['train/box_loss'], 
                       label=f'{name} (Train)', color=color, linestyle='-', alpha=0.7)
        axes[0, 1].plot(df.index + 1, df['val/box_loss'], 
                       label=f'{name} (Val)', color=color, linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Box Loss Progression')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Box Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[1, 0].scatter(df['metrics/recall(B)'], df['metrics/precision(B)'], 
                          label=name, color=color, alpha=0.6, s=20)
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(df.index + 1, df['lr/pg0'], 
                       label=name, color=color, linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    
    print("🔍 YOLO TRAINING RESULTS ANALYZER")
    print("=" * 50)
    
    # Find all experiment directories
    runs_dir = Path('./runs')
    experiment_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and 'aquarium' in d.name]
    
    if not experiment_dirs:
        print("❌ No experiment directories found in ./runs")
        return
    
    results_list = []
    experiment_names = []
    
    for exp_dir in sorted(experiment_dirs):
        exp_name = exp_dir.name
        print(f"\n🔍 Analyzing: {exp_name}")
        results = analyze_training_results(exp_dir, exp_name)
        results_list.append(results)
        experiment_names.append(exp_name)
    
    # Create comparison plot
    if len([r for r in results_list if r is not None]) > 0:
        print(f"\n📈 Creating comparison plot...")
        plot_training_comparison(results_list, experiment_names)
    
if __name__ == "__main__":
    main()
