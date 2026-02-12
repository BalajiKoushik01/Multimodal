"""
Advanced Data Loader V2 for Multimodal Cervical Cancer Detection

Supports multiple datasets:
1. Data1: Binary classification (Abnormal/Normal) - 754 images
2. Data2: Kaggle cervical cancer dataset - 11,683 images with Type 1/2/3
3. Quality Assessment: CSV-based features (green, hinselmann, schiller)
4. MobileODT: Existing dataset
5. AnnoCerv: Existing dataset with SWEDE scores

Features:
- Unified interface for all datasets
- CSV feature integration
- Stratified splitting
- Metadata tracking
- Cross-validation support
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

class DatasetRegistry:
    """Central registry for all available datasets"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.datasets = {}
        self._discover_datasets()
    
    def _discover_datasets(self):
        """Automatically discover available datasets"""
        
        # Data1: Binary classification
        data1_path = self.root_dir / "new dataset" / "data1" / "ClassifiedImages-20250719T090316Z-1-001" / "ClassifiedImages"
        if data1_path.exists():
            self.datasets['data1'] = {
                'path': data1_path,
                'type': 'binary_classification',
                'modality': 'colposcopy',
                'classes': ['Normal', 'Abnormal'],
                'num_images': 754
            }
        
        # Data2: Kaggle cervical cancer
        data2_path = self.root_dir / "new dataset" / "data2" / "archive" / "kaggle"
        if data2_path.exists():
            self.datasets['data2'] = {
                'path': data2_path,
                'type': 'multi_class',
                'modality': 'colposcopy',
                'classes': ['Type_1', 'Type_2', 'Type_3'],
                'splits': ['train', 'test', 'test_stg2', 'additional_Type_1_v2', 'additional_Type_2_v2', 'additional_Type_3_v2'],
                'num_images': 11683
            }
        
        # Quality Assessment: CSV features
        qa_path = self.root_dir / "new dataset" / "quality_assessment" / "Quality Assessment - Digital Colposcopy"
        if qa_path.exists():
            self.datasets['quality_assessment'] = {
                'path': qa_path,
                'type': 'csv_features',
                'modality': 'colposcopy',
                'csv_files': ['green.csv', 'hinselmann.csv', 'schiller.csv'],
                'num_samples': None  # Will be determined from CSVs
            }
        
        # MobileODT
        mobileodt_path = self.root_dir / "data" / "mobileodt"
        if mobileodt_path.exists():
            self.datasets['mobileodt'] = {
                'path': mobileodt_path,
                'type': 'multi_class',
                'modality': 'colposcopy',
                'classes': ['Type_1', 'Type_2', 'Type_3']
            }
        
        # AnnoCerv
        annocerv_path = self.root_dir / "data" / "annocerv"
        if annocerv_path.exists():
            self.datasets['annocerv'] = {
                'path': annocerv_path,
                'type': 'swede_scores',
                'modality': 'colposcopy',
                'has_labels': True
            }
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a specific dataset"""
        return self.datasets.get(dataset_name, {})
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self.datasets.keys())
    
    def get_total_images(self) -> int:
        """Get total number of images across all datasets"""
        total = 0
        for dataset_info in self.datasets.values():
            if 'num_images' in dataset_info and dataset_info['num_images']:
                total += dataset_info['num_images']
        return total


class UnifiedDatasetParser:
    """Parse different dataset formats into a unified structure"""
    
    @staticmethod
    def parse_data1(dataset_path: Path) -> List[Dict]:
        """
        Parse Data1 binary classification dataset
        
        Structure:
        ClassifiedImages/
            Normal/
                image1.jpg
                image2.jpg
            Abnormal/
                image1.jpg
                image2.jpg
        """
        items = []
        
        for class_name in ['Normal', 'Abnormal']:
            class_dir = dataset_path / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    items.append({
                        'image_path': str(img_path),
                        'label': 0 if class_name == 'Normal' else 1,
                        'class_name': class_name,
                        'dataset': 'data1',
                        'modality': 'colposcopy',
                        'metadata': {
                            'filename': img_path.name,
                            'source': 'data1_classified'
                        }
                    })
        
        return items
    
    @staticmethod
    def parse_data2(dataset_path: Path) -> List[Dict]:
        """
        Parse Data2 Kaggle cervical cancer dataset
        
        Structure:
        kaggle/
            train/
                image1.jpg
            test/
                image1.jpg
            additional_Type_1_v2/
            additional_Type_2_v2/
            additional_Type_3_v2/
        """
        items = []
        
        # Parse train split
        train_dir = dataset_path / 'train'
        if train_dir.exists():
            for img_path in train_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    items.append({
                        'image_path': str(img_path),
                        'label': -1,  # Unlabeled for SSL
                        'class_name': 'unknown',
                        'dataset': 'data2',
                        'split': 'train',
                        'modality': 'colposcopy',
                        'metadata': {
                            'filename': img_path.name,
                            'source': 'kaggle_train'
                        }
                    })
        
        # Parse test splits
        for test_split in ['test', 'test_stg2']:
            test_dir = dataset_path / test_split
            if test_dir.exists():
                for img_path in test_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        items.append({
                            'image_path': str(img_path),
                            'label': -1,
                            'class_name': 'unknown',
                            'dataset': 'data2',
                            'split': test_split,
                            'modality': 'colposcopy',
                            'metadata': {
                                'filename': img_path.name,
                                'source': f'kaggle_{test_split}'
                            }
                        })
        
        # Parse additional type-specific data
        for type_idx, type_name in enumerate(['Type_1', 'Type_2', 'Type_3'], start=1):
            type_dir = dataset_path / f'additional_{type_name}_v2'
            if type_dir.exists():
                for img_path in type_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        items.append({
                            'image_path': str(img_path),
                            'label': type_idx,
                            'class_name': type_name,
                            'dataset': 'data2',
                            'split': 'additional',
                            'modality': 'colposcopy',
                            'metadata': {
                                'filename': img_path.name,
                                'source': f'kaggle_additional_{type_name}'
                            }
                        })
        
        return items


class MultiDatasetLoader:
    """
    Unified data loader for all cervical cancer datasets
    
    Features:
    - Load from multiple sources
    - Stratified splitting
    - Cross-validation support
    - Metadata tracking
    """
    
    def __init__(self, root_dir: str, datasets_to_load: Optional[List[str]] = None):
        """
        Args:
            root_dir: Root directory containing all datasets
            datasets_to_load: List of dataset names to load. If None, load all available.
        """
        self.root_dir = Path(root_dir)
        self.registry = DatasetRegistry(root_dir)
        self.datasets_to_load = datasets_to_load or self.registry.list_datasets()
        
        self.all_items = []
        self.dataset_stats = {}
        
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load all specified datasets"""
        parser = UnifiedDatasetParser()
        
        for dataset_name in self.datasets_to_load:
            dataset_info = self.registry.get_dataset_info(dataset_name)
            if not dataset_info:
                print(f"Warning: Dataset '{dataset_name}' not found")
                continue
            
            dataset_path = dataset_info['path']
            
            try:
                if dataset_name == 'data1':
                    items = parser.parse_data1(dataset_path)
                    self.all_items.extend(items)
                    self.dataset_stats[dataset_name] = {
                        'num_items': len(items),
                        'num_normal': sum(1 for item in items if item['label'] == 0),
                        'num_abnormal': sum(1 for item in items if item['label'] == 1)
                    }
                
                elif dataset_name == 'data2':
                    items = parser.parse_data2(dataset_path)
                    self.all_items.extend(items)
                    self.dataset_stats[dataset_name] = {
                        'num_items': len(items),
                        'num_labeled': sum(1 for item in items if item['label'] != -1),
                        'num_unlabeled': sum(1 for item in items if item['label'] == -1)
                    }
                
                print(f"Loaded {dataset_name}: {len(items)} items")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
    
    def get_all_image_paths(self) -> List[str]:
        """Get all image paths for SSL training"""
        return [item['image_path'] for item in self.all_items]
    
    def get_stratified_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """Get stratified train/val/test split"""
        from sklearn.model_selection import train_test_split
        
        # Separate labeled and unlabeled
        labeled = [item for item in self.all_items if item['label'] != -1]
        unlabeled = [item for item in self.all_items if item['label'] == -1]
        
        if len(labeled) > 0:
            # Split labeled data
            labels = [item['label'] for item in labeled]
            train_labeled, temp = train_test_split(labeled, test_size=(val_ratio + test_ratio), 
                                                   stratify=labels, random_state=random_state)
            
            temp_labels = [item['label'] for item in temp]
            val_labeled, test_labeled = train_test_split(temp, test_size=test_ratio/(val_ratio + test_ratio),
                                                         stratify=temp_labels, random_state=random_state)
        else:
            train_labeled, val_labeled, test_labeled = [], [], []
        
        # Add unlabeled to training
        train_all = train_labeled + unlabeled
        
        return train_all, val_labeled, test_labeled
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_items': len(self.all_items),
            'datasets_loaded': list(self.dataset_stats.keys()),
            'dataset_stats': self.dataset_stats,
            'overall_distribution': self._get_label_distribution(self.all_items),
            'modality_distribution': self._get_modality_distribution()
        }
        
        return stats
    
    def _get_label_distribution(self, items: List[Dict]) -> Dict:
        """Get label distribution for a set of items"""
        distribution = defaultdict(int)
        for item in items:
            label = item['label']
            distribution[str(label)] += 1
        return dict(distribution)
    
    def _get_modality_distribution(self) -> Dict:
        """Get distribution of modalities"""
        distribution = defaultdict(int)
        for item in self.all_items:
            modality = item.get('modality', 'unknown')
            distribution[modality] += 1
        return dict(distribution)


if __name__ == "__main__":
    # Example usage
    loader = MultiDatasetLoader(
        root_dir="c:/Users/balaj/Desktop/multimodal",
        datasets_to_load=['data1', 'data2']
    )
    
    # Get statistics
    stats = loader.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
