#!/usr/bin/env python3
"""
Dataset Explorer for Fish Classification Project
This script provides quick insights into the dataset structure and statistics.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def explore_dataset():
    """Explore the fish dataset structure and statistics"""
    
    base_dir = './Dataset'
    splits = ['train', 'val', 'test']
    
    if not os.path.exists(base_dir):
        print("❌ Dataset directory not found!")
        return
    
    print("🐟 Fish Dataset Explorer")
    print("=" * 50)
    
    # Get class names from train directory
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        print("❌ Training directory not found!")
        return
    
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"📊 Found {len(class_names)} fish classes:")
    for i, class_name in enumerate(class_names, 1):
        print(f"   {i:2d}. {class_name}")
    
    print(f"\n📁 Dataset splits: {', '.join(splits)}")
    
    # Count images in each split and class
    data = defaultdict(dict)
    total_images = 0
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  {split} directory not found!")
            continue
            
        split_total = 0
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                # Count image files
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                count = len(image_files)
                data[class_name][split] = count
                split_total += count
            else:
                data[class_name][split] = 0
        
        print(f"   {split.capitalize()}: {split_total:,} images")
        total_images += split_total
    
    print(f"\n📈 Total images: {total_images:,}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(data).T
    df = df.fillna(0).astype(int)
    
    # Add totals
    df['Total'] = df.sum(axis=1)
    
    print(f"\n📋 Detailed breakdown:")
    print(df.to_string())
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Average images per class: {df['Total'].mean():.1f}")
    print(f"   Min images per class: {df['Total'].min()}")
    print(f"   Max images per class: {df['Total'].max()}")
    print(f"   Standard deviation: {df['Total'].std():.1f}")
    
    # Check for class imbalance
    min_count = df['Total'].min()
    max_count = df['Total'].max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 2.0:
        print(f"⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        print("   Consider data augmentation or class balancing techniques")
    else:
        print(f"✅ Classes are relatively balanced (ratio: {imbalance_ratio:.2f})")
    
    # Visualization
    try:
        print(f"\n📊 Generating visualizations...")
        
        # Plot 1: Images per split
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        split_totals = df[splits].sum()
        ax1.pie(split_totals.values, labels=split_totals.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Dataset Distribution by Split')
        
        # Plot 2: Images per class
        df[splits].plot(kind='bar', ax=ax2, stacked=True)
        ax2.set_title('Images per Class by Split')
        ax2.set_xlabel('Fish Classes')
        ax2.set_ylabel('Number of Images')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Analysis saved as 'dataset_analysis.png'")
        
    except ImportError:
        print("📊 Matplotlib not available for visualization")
    except Exception as e:
        print(f"⚠️  Error creating visualization: {e}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if total_images < 1000:
        print("   • Consider data augmentation to increase dataset size")
    if imbalance_ratio > 2.0:
        print("   • Use class weights or balanced sampling during training")
    print("   • Ensure image quality is consistent across classes")
    print("   • Consider using transfer learning for better performance")
    
    return df

def check_image_formats():
    """Check image formats and potential issues"""
    print(f"\n🔍 Checking image formats...")
    
    base_dir = './Dataset'
    formats = defaultdict(int)
    issues = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                ext = file.lower().split('.')[-1]
                formats[ext] += 1
            elif '.' in file:
                issues.append(f"Unknown format: {file} in {root}")
    
    print("📁 Image formats found:")
    for fmt, count in sorted(formats.items()):
        print(f"   .{fmt}: {count:,} files")
    
    if issues:
        print(f"\n⚠️  Potential issues ({len(issues)} files):")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"   {issue}")
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more")

if __name__ == "__main__":
    df = explore_dataset()
    check_image_formats()
    
    print(f"\n✅ Dataset exploration complete!")
    print(f"💡 Use this information to optimize your model training strategy.")
