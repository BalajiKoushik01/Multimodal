"""
SSL Pre-Training Script

Executable script to start self-supervised pre-training on cervical cancer datasets.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json

from training.ssl_trainer import SSLTrainer
from data.data_loader_v2 import MultiDatasetLoader
from data.advanced_augmentation import get_colposcopy_augmentation


class SSLDataset(Dataset):
    """
    Dataset for SSL pre-training
    
    Returns pairs of augmented views for contrastive learning.
    """
    
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths: List of image file paths
            transform: Augmentation transform
        """
        self.image_paths = image_paths
        self.transform = transform or get_colposcopy_augmentation(mode='train')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation twice for contrastive learning
        # For simplicity, using same modality twice (colpo = histo in this case)
        view1 = self.transform(image)
        view2 = self.transform(image)
        
        return {
            'colposcopy': view1,
            'histopathology': view2  # Same image, different augmentation
        }


def main():
    """Main training function"""
    
    print("=" * 70)
    print("CERVICAL CANCER DETECTION - SSL PRE-TRAINING")
    print("=" * 70)
    
    # Configuration
    config = {
        # Model architecture
        'backbone': 'resnet50',  # Start with ResNet50 (ViT requires more memory)
        'projection_dim': 2048,
        'ssl_projection_dim': 128,
        'pretrained': True,
        'use_cross_attention': True,
        'num_attention_heads': 8,
        'num_fusion_layers': 2,
        'num_projection_layers': 3,
        'model_params': {
            'projection_head_hidden_dim': 4096,
            'num_classes': 2,
            'severity_classes': 4
        },
        
        # Training strategy
        'ssl_strategy': 'multimodal',  # Multimodal contrastive learning
        'num_epochs': 50,  # Start with 50 epochs for initial run
        'batch_size': 16,  # Reduced for memory constraints
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'temperature': 0.07,
        
        # Optimization
        'use_amp': True,  # Mixed precision
        'accumulation_steps': 4,  # Effective batch size = 64
        'warmup_epochs': 5,
        'use_multi_gpu': torch.cuda.device_count() > 1,
        
        # Checkpointing
        'output_dir': str(project_root / 'outputs' / 'ssl_pretraining'),
        'save_freq': 5,
        
        # Data
        'image_size': 224,
        'num_workers': 4
    }
    
    # Print configuration
    print("\nTraining Configuration:")
    print(json.dumps(config, indent=2))
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("\n⚠ No GPU available, training will be slow on CPU")
    
    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)
    
    try:
        loader = MultiDatasetLoader(
            root_dir=str(project_root),
            datasets_to_load=['data1', 'data2']
        )
        
        # Get all images for SSL (including unlabeled)
        all_images = loader.get_all_image_paths()
        print(f"\nTotal images for SSL: {len(all_images)}")
        
        # Split into train/val (90/10 for SSL)
        split_idx = int(0.9 * len(all_images))
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")
        
    except Exception as e:
        print(f"\n⚠ Error loading datasets: {e}")
        print("Using dummy dataset for testing...")
        
        # Create dummy dataset for testing
        train_images = [str(project_root / 'data' / 'dummy.jpg')] * 100
        val_images = [str(project_root / 'data' / 'dummy.jpg')] * 20
    
    # Create datasets
    print("\nCreating data loaders...")
    train_dataset = SSLDataset(train_images)
    val_dataset = SSLDataset(val_images)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize trainer
    print("\n" + "=" * 70)
    print("INITIALIZING TRAINER")
    print("=" * 70)
    
    trainer = SSLTrainer(config)
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    try:
        trainer.train(train_loader, val_loader)
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nBest validation loss: {trainer.best_loss:.4f}")
        print(f"Checkpoints saved to: {config['output_dir']}/checkpoints")
        print(f"Logs saved to: {config['output_dir']}/logs")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved!")
        
    except Exception as e:
        print(f"\n\n⚠ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved!")


if __name__ == "__main__":
    main()
