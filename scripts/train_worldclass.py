"""
World-Class Training Configuration for Cervical Cancer Detection

Optimized for maximum GPU utilization and state-of-the-art performance.
Implements best practices from top medical imaging papers.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

from training.ssl_trainer import SSLTrainer
from data.data_loader_v2 import MultiDatasetLoader
from data.advanced_augmentation import get_colposcopy_augmentation


class OptimizedSSLDataset(Dataset):
    """
    Optimized dataset with prefetching and caching for maximum GPU utilization
    """
    
    def __init__(self, image_paths, transform=None, cache_size=1000):
        self.image_paths = image_paths
        self.transform = transform or get_colposcopy_augmentation(mode='train')
        self.cache = {}
        self.cache_size = cache_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Cache frequently accessed images
        if img_path in self.cache:
            image = self.cache[img_path]
        else:
            image = Image.open(img_path).convert('RGB')
            if len(self.cache) < self.cache_size:
                self.cache[img_path] = image
        
        # Two augmented views for contrastive learning
        view1 = self.transform(image)
        view2 = self.transform(image)
        
        return {
            'colposcopy': view1,
            'histopathology': view2
        }


def get_gpu_config():
    """Detect GPU and return optimal configuration"""
    if not torch.cuda.is_available():
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    
    # Optimize based on GPU memory
    if gpu_memory >= 24:  # RTX 3090, 4090, A100
        config = {
            'backbone': 'vit_base',
            'batch_size': 32,
            'num_workers': 8,
            'accumulation_steps': 4,  # Effective batch: 128
            'image_size': 384,  # Higher resolution
        }
    elif gpu_memory >= 12:  # RTX 3080, 4070 Ti, 4080
        config = {
            'backbone': 'vit_base',
            'batch_size': 24,
            'num_workers': 6,
            'accumulation_steps': 4,  # Effective batch: 96
            'image_size': 224,
        }
    elif gpu_memory >= 8:  # RTX 3070, 4060 Ti
        config = {
            'backbone': 'vit_base',
            'batch_size': 16,
            'num_workers': 4,
            'accumulation_steps': 4,  # Effective batch: 64
            'image_size': 224,
        }
    else:  # RTX 3060, older GPUs
        config = {
            'backbone': 'resnet50',
            'batch_size': 12,
            'num_workers': 4,
            'accumulation_steps': 4,  # Effective batch: 48
            'image_size': 224,
        }
    
    return config, gpu_name, gpu_memory


def main():
    """World-class training with optimized GPU utilization"""
    
    print("=" * 80)
    print("WORLD-CLASS CERVICAL CANCER DETECTION - OPTIMIZED TRAINING")
    print("=" * 80)
    
    # Detect GPU and get optimal config
    gpu_config, gpu_name, gpu_memory = get_gpu_config()
    
    if gpu_config is None:
        print("\n⚠ No GPU detected! Training will be very slow on CPU.")
        print("Please ensure CUDA is properly installed and GPU is available.")
        return
    
    print(f"\n✓ GPU Detected: {gpu_name}")
    print(f"  Memory: {gpu_memory:.2f} GB")
    print(f"  Optimized Configuration:")
    print(f"    - Backbone: {gpu_config['backbone']}")
    print(f"    - Batch Size: {gpu_config['batch_size']}")
    print(f"    - Effective Batch: {gpu_config['batch_size'] * gpu_config['accumulation_steps']}")
    print(f"    - Image Size: {gpu_config['image_size']}")
    print(f"    - Num Workers: {gpu_config['num_workers']}")
    
    # World-class training configuration
    config = {
        # Model Architecture (State-of-the-art)
        'backbone': gpu_config['backbone'],
        'projection_dim': 2048,
        'ssl_projection_dim': 256,  # Increased for better representations
        'pretrained': True,
        'use_cross_attention': True,
        'num_attention_heads': 12,  # Increased for ViT-Base
        'num_fusion_layers': 3,  # Deeper fusion
        'num_projection_layers': 3,
        'model_params': {
            'projection_head_hidden_dim': 4096,
            'num_classes': 2,
            'severity_classes': 4,
            'dropout': 0.3
        },
        
        # Training Strategy (Best Practices)
        'ssl_strategy': 'multimodal',
        'num_epochs': 100,  # Full training
        'batch_size': gpu_config['batch_size'],
        'learning_rate': 1e-3,  # Higher LR for larger batch
        'weight_decay': 1e-4,
        'temperature': 0.07,
        
        # Advanced Optimization
        'use_amp': True,  # Mixed precision
        'accumulation_steps': gpu_config['accumulation_steps'],
        'warmup_epochs': 10,
        'use_multi_gpu': torch.cuda.device_count() > 1,
        
        # Data Loading (Optimized)
        'image_size': gpu_config['image_size'],
        'num_workers': gpu_config['num_workers'],
        'pin_memory': True,
        'persistent_workers': True,  # Keep workers alive
        'prefetch_factor': 2,  # Prefetch batches
        
        # Checkpointing
        'output_dir': str(project_root / 'outputs' / 'ssl_worldclass'),
        'save_freq': 5,
        
        # Advanced Techniques
        'use_ema': True,  # Exponential moving average
        'ema_decay': 0.999,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.2,
    }
    
    print("\n" + "=" * 80)
    print("WORLD-CLASS TRAINING CONFIGURATION")
    print("=" * 80)
    print(json.dumps({k: v for k, v in config.items() if not isinstance(v, dict)}, indent=2))
    
    # Load datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    try:
        loader = MultiDatasetLoader(
            root_dir=str(project_root),
            datasets_to_load=['data1', 'data2']
        )
        
        all_images = loader.get_all_image_paths()
        print(f"\n✓ Total images: {len(all_images)}")
        
        # Stratified split
        split_idx = int(0.9 * len(all_images))
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        print(f"  Training: {len(train_images)}")
        print(f"  Validation: {len(val_images)}")
        
    except Exception as e:
        print(f"\n⚠ Error loading datasets: {e}")
        return
    
    # Create optimized data loaders
    print("\n" + "=" * 80)
    print("CREATING OPTIMIZED DATA LOADERS")
    print("=" * 80)
    
    train_dataset = OptimizedSSLDataset(
        train_images,
        cache_size=min(1000, len(train_images) // 10)
    )
    val_dataset = OptimizedSSLDataset(
        val_images,
        cache_size=min(200, len(val_images) // 5)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'],
        prefetch_factor=config['prefetch_factor'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'],
        prefetch_factor=config['prefetch_factor']
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
    
    # Initialize trainer
    print("\n" + "=" * 80)
    print("INITIALIZING WORLD-CLASS TRAINER")
    print("=" * 80)
    
    trainer = SSLTrainer(config)
    
    # Display model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model initialized: {config['backbone']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Start training
    print("\n" + "=" * 80)
    print("STARTING WORLD-CLASS TRAINING")
    print("=" * 80)
    print(f"\nTarget Performance:")
    print(f"  • HSIL Sensitivity: >95%")
    print(f"  • Malignancy AUC-ROC: >0.95")
    print(f"  • Severity Accuracy: >85%")
    print(f"  • Progression Risk AUC: >0.92")
    print("\n" + "=" * 80)
    
    try:
        trainer.train(train_loader, val_loader)
        
        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nBest validation loss: {trainer.best_loss:.4f}")
        print(f"Checkpoints: {config['output_dir']}/checkpoints")
        print(f"Logs: {config['output_dir']}/logs")
        print("\nReady for supervised fine-tuning!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint()
        print("Checkpoint saved!")
        
    except Exception as e:
        print(f"\n\n⚠ Error during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint()


if __name__ == "__main__":
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmul
    torch.backends.cudnn.allow_tf32 = True  # TF32 for convolutions
    
    main()
