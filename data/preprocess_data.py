"""
Data Preprocessing Pipeline for Cervical Cancer Detection

Comprehensive preprocessing including:
1. Image quality assessment and filtering
2. Dataset statistics and distribution analysis
3. Data cleaning and validation
4. Preprocessing for training (normalization, resizing)
5. Dataset balancing strategies
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import cv2


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    """
    
    def __init__(self, root_dir, output_dir=None):
        """
        Args:
            root_dir: Root directory containing datasets
            output_dir: Directory to save preprocessing results
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / "preprocessed"
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = defaultdict(dict)
        self.quality_report = []
    
    def assess_image_quality(self, image_path, min_size=100, max_size=5000):
        """
        Assess image quality
        
        Checks:
        - Image can be loaded
        - Image size is within acceptable range
        - Image is not corrupted
        - Image has sufficient variance (not blank)
        
        Returns:
            dict: Quality metrics
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Check dimensions
            h, w = img_array.shape[:2]
            if h < min_size or w < min_size:
                return {'valid': False, 'reason': 'too_small', 'size': (h, w)}
            if h > max_size or w > max_size:
                return {'valid': False, 'reason': 'too_large', 'size': (h, w)}
            
            # Check if grayscale or RGB
            if len(img_array.shape) == 2:
                channels = 1
            elif len(img_array.shape) == 3:
                channels = img_array.shape[2]
            else:
                return {'valid': False, 'reason': 'invalid_shape', 'shape': img_array.shape}
            
            # Check variance (detect blank images)
            variance = np.var(img_array)
            if variance < 10:  # Very low variance = likely blank
                return {'valid': False, 'reason': 'low_variance', 'variance': variance}
            
            # Check for corruption (NaN or Inf values)
            if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                return {'valid': False, 'reason': 'corrupted', 'has_nan': True}
            
            # Compute quality metrics
            metrics = {
                'valid': True,
                'size': (h, w),
                'channels': channels,
                'variance': float(variance),
                'mean': float(np.mean(img_array)),
                'std': float(np.std(img_array)),
                'min': int(np.min(img_array)),
                'max': int(np.max(img_array))
            }
            
            return metrics
            
        except Exception as e:
            return {'valid': False, 'reason': 'load_error', 'error': str(e)}
    
    def analyze_dataset(self, dataset_path, dataset_name):
        """
        Analyze a single dataset
        
        Args:
            dataset_path: Path to dataset directory
            dataset_name: Name of the dataset
        
        Returns:
            dict: Dataset statistics
        """
        print(f"\nAnalyzing {dataset_name}...")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"  Warning: {dataset_path} does not exist")
            return None
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
            image_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
        
        print(f"  Found {len(image_files)} images")
        
        # Analyze each image
        valid_images = []
        invalid_images = []
        size_distribution = []
        variance_distribution = []
        
        for img_path in tqdm(image_files, desc=f"  Processing {dataset_name}"):
            quality = self.assess_image_quality(img_path)
            
            if quality['valid']:
                valid_images.append(str(img_path))
                size_distribution.append(quality['size'])
                variance_distribution.append(quality['variance'])
            else:
                invalid_images.append({
                    'path': str(img_path),
                    'reason': quality.get('reason', 'unknown')
                })
                self.quality_report.append({
                    'dataset': dataset_name,
                    'path': str(img_path),
                    'reason': quality.get('reason', 'unknown')
                })
        
        # Compute statistics
        stats = {
            'dataset_name': dataset_name,
            'total_images': len(image_files),
            'valid_images': len(valid_images),
            'invalid_images': len(invalid_images),
            'validity_rate': len(valid_images) / len(image_files) if len(image_files) > 0 else 0,
            'size_distribution': {
                'mean_height': float(np.mean([s[0] for s in size_distribution])) if size_distribution else 0,
                'mean_width': float(np.mean([s[1] for s in size_distribution])) if size_distribution else 0,
                'min_height': int(np.min([s[0] for s in size_distribution])) if size_distribution else 0,
                'max_height': int(np.max([s[0] for s in size_distribution])) if size_distribution else 0,
                'min_width': int(np.min([s[1] for s in size_distribution])) if size_distribution else 0,
                'max_width': int(np.max([s[1] for s in size_distribution])) if size_distribution else 0,
            },
            'variance_distribution': {
                'mean': float(np.mean(variance_distribution)) if variance_distribution else 0,
                'std': float(np.std(variance_distribution)) if variance_distribution else 0,
                'min': float(np.min(variance_distribution)) if variance_distribution else 0,
                'max': float(np.max(variance_distribution)) if variance_distribution else 0,
            },
            'invalid_reasons': {}
        }
        
        # Count invalid reasons
        for inv in invalid_images:
            reason = inv['reason']
            stats['invalid_reasons'][reason] = stats['invalid_reasons'].get(reason, 0) + 1
        
        self.stats[dataset_name] = stats
        
        if len(image_files) > 0:
            print(f"  Valid: {len(valid_images)}/{len(image_files)} ({stats['validity_rate']*100:.1f}%)")
        else:
            print(f"  No images found in dataset")
        
        if invalid_images:
            print(f"  Invalid reasons: {stats['invalid_reasons']}")
        
        return stats
    
    def analyze_all_datasets(self):
        """
        Analyze all datasets in the root directory
        """
        print("=" * 60)
        print("DATASET QUALITY ANALYSIS")
        print("=" * 60)
        
        # Known datasets
        datasets = {
            'data1': self.root_dir / 'data' / 'data1',
            'data2': self.root_dir / 'data' / 'data2',
            'MobileODT': self.root_dir / 'data' / 'MobileODT',
            'AnnoCerv': self.root_dir / 'data' / 'AnnoCerv'
        }
        
        all_stats = {}
        for name, path in datasets.items():
            stats = self.analyze_dataset(path, name)
            if stats:
                all_stats[name] = stats
        
        # Save statistics
        stats_file = self.output_dir / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")
        
        # Save quality report
        if self.quality_report:
            report_file = self.output_dir / 'quality_report.json'
            with open(report_file, 'w') as f:
                json.dump(self.quality_report, f, indent=2)
            print(f"Quality report saved to: {report_file}")
        
        return all_stats
    
    def generate_summary_report(self):
        """
        Generate a summary report of all datasets
        """
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        
        total_images = 0
        total_valid = 0
        
        for dataset_name, stats in self.stats.items():
            print(f"\n{dataset_name}:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Valid images: {stats['valid_images']}")
            print(f"  Validity rate: {stats['validity_rate']*100:.1f}%")
            print(f"  Avg size: {stats['size_distribution']['mean_height']:.0f} × {stats['size_distribution']['mean_width']:.0f}")
            
            total_images += stats['total_images']
            total_valid += stats['valid_images']
        
        print("\n" + "-" * 60)
        print(f"TOTAL ACROSS ALL DATASETS:")
        print(f"  Total images: {total_images}")
        print(f"  Valid images: {total_valid}")
        print(f"  Overall validity rate: {total_valid/total_images*100:.1f}%")
        print("=" * 60)
    
    def visualize_distributions(self):
        """
        Create visualization of dataset distributions
        """
        print("\nGenerating visualizations...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Dataset sizes
        ax = axes[0, 0]
        datasets = list(self.stats.keys())
        valid_counts = [self.stats[d]['valid_images'] for d in datasets]
        invalid_counts = [self.stats[d]['invalid_images'] for d in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, valid_counts, width, label='Valid', color='green', alpha=0.7)
        ax.bar(x + width/2, invalid_counts, width, label='Invalid', color='red', alpha=0.7)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Number of Images')
        ax.set_title('Dataset Sizes (Valid vs Invalid)')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Validity rates
        ax = axes[0, 1]
        validity_rates = [self.stats[d]['validity_rate'] * 100 for d in datasets]
        colors = ['green' if v > 95 else 'orange' if v > 90 else 'red' for v in validity_rates]
        ax.bar(datasets, validity_rates, color=colors, alpha=0.7)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Validity Rate (%)')
        ax.set_title('Dataset Quality (Validity Rate)')
        ax.set_ylim([0, 105])
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Image size distribution
        ax = axes[1, 0]
        for dataset in datasets:
            mean_h = self.stats[dataset]['size_distribution']['mean_height']
            mean_w = self.stats[dataset]['size_distribution']['mean_width']
            ax.scatter(mean_w, mean_h, s=200, alpha=0.6, label=dataset)
        ax.set_xlabel('Mean Width (pixels)')
        ax.set_ylabel('Mean Height (pixels)')
        ax.set_title('Average Image Dimensions')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Variance distribution
        ax = axes[1, 1]
        variance_means = [self.stats[d]['variance_distribution']['mean'] for d in datasets]
        variance_stds = [self.stats[d]['variance_distribution']['std'] for d in datasets]
        ax.bar(datasets, variance_means, yerr=variance_stds, capsize=5, alpha=0.7, color='blue')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Image Variance')
        ax.set_title('Image Variance Distribution (Mean ± Std)')
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        viz_file = self.output_dir / 'dataset_distributions.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {viz_file}")
        plt.close()


if __name__ == "__main__":
    # Run preprocessing
    print("Starting data preprocessing pipeline...")
    
    root_dir = r"c:\Users\balaj\Desktop\multimodal"
    preprocessor = DataPreprocessor(root_dir)
    
    # Analyze all datasets
    stats = preprocessor.analyze_all_datasets()
    
    # Generate summary
    preprocessor.generate_summary_report()
    
    # Create visualizations
    preprocessor.visualize_distributions()
    
    print("\n✓ Preprocessing complete!")
    print(f"Results saved to: {preprocessor.output_dir}")
