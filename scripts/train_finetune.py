
import os
import argparse
import sys

# Add parent directory to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CervicalDataset
from models.dual_encoder import DualEncoder

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])

    # Dataset & Dataloader
    # 1. Standard mock/flat datasets
    ds_colpo_std = CervicalDataset(config['dataset_path'], modality='colposcopy', transform=transform, mode='train')
    
    # 2. External datasets
    annocerv_path = os.path.join(config['dataset_path'], 'annocerv_real')
    mobileodt_path = os.path.join(config['dataset_path'], 'mobileodt')
    
    ds_annocerv = CervicalDataset(annocerv_path, modality='annocerv', transform=transform, mode='train')
    ds_mobileodt = CervicalDataset(mobileodt_path, modality='mobileodt', transform=transform, mode='train')
    
    # Combine all colposcopy sources
    datasets_to_combine = []
    if len(ds_colpo_std) > 0: datasets_to_combine.append(ds_colpo_std)
    if len(ds_annocerv) > 0: datasets_to_combine.append(ds_annocerv)
    if len(ds_mobileodt) > 0: datasets_to_combine.append(ds_mobileodt)
    
    if not datasets_to_combine:
        print("Error: No colposcopy images found.")
        return

    from data.dataset import CombinedDataset
    dataset_colpo = CombinedDataset(datasets_to_combine)
    print(f"Fine-tuning on {len(dataset_colpo)} colposcopy images.")

    # Histopathology (standard/mock)
    dataset_histo = CervicalDataset(config['dataset_path'], modality='histopathology', transform=transform, mode='train')
    
    # DataLoaders
    # Note: drop_last=True to avoid shape mismatch if cycling
    loader_colpo = DataLoader(dataset_colpo, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    loader_histo = DataLoader(dataset_histo, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    
    # Model
    model = DualEncoder(config).to(device)
    
    # Load Pretrained Weights
    if os.path.exists(config['weights_save_path']):
         checkpoints = [f for f in os.listdir(config['weights_save_path']) if f.endswith('.pth')]
         if checkpoints:
             # Sort by creation time to get the latest
             latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(config['weights_save_path'], x)))
             checkpoint_path = os.path.join(config['weights_save_path'], latest_checkpoint)
             print(f"Loading SSL weights from {checkpoint_path}")
             model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    # Handle cycling if datasets are different sizes
    max_len = max(len(loader_colpo), len(loader_histo))
    from itertools import cycle
    if len(loader_colpo) < max_len: loader_colpo = cycle(loader_colpo)
    if len(loader_histo) < max_len: loader_histo = cycle(loader_histo)

    # Optimizer
    optimizer = optim.Adam([
        {'params': model.colpo_encoder.parameters(), 'lr': config['learning_rate'] * 0.1},
        {'params': model.histo_encoder.parameters(), 'lr': config['learning_rate'] * 0.1},
        {'params': model.malignancy_head.parameters(), 'lr': config['learning_rate']},
        {'params': model.severity_head.parameters(), 'lr': config['learning_rate']}
    ], lr=config['learning_rate'])
    
    # Losses
    criterion_cls = nn.CrossEntropyLoss().to(device)

    # Training Loop
    model.train()
    print("Starting fine-tuning...")
    
    for epoch in range(config['max_epochs']):
        epoch_loss = 0.0
        
        progress_bar = tqdm(zip(loader_colpo, loader_histo), total=max_len, desc=f"Epoch {epoch+1}/{config['max_epochs']}")
        
        for batch_idx, (data_c, data_h) in enumerate(progress_bar):
            # Unpack colposcopy data
            # It could be (img, label) or just img depending on dataset implementation
            if isinstance(data_c, list) or isinstance(data_c, tuple):
                img_c, label_c = data_c
            else:
                img_c = data_c
                label_c = torch.full((img_c.size(0),), -1) # No label

            # Unpack histopathology data
            if isinstance(data_h, list) or isinstance(data_h, tuple):
                img_h, label_h = data_h
            else:
                img_h = data_h
            
            img_c = img_c.to(device)
            img_h = img_h.to(device)
            
            # Prepare Labels
            target_mal = torch.randint(0, config['model_params']['num_classes'], (img_c.size(0),)).to(device)
            
            # Overwrite with real labels where available
            label_c = label_c.to(device)
            mask_labeled = (label_c != -1)
            if mask_labeled.any():
                target_mal[mask_labeled] = label_c[mask_labeled].long()
            
            target_sev = torch.randint(0, config['model_params']['severity_classes'], (img_c.size(0),)).to(device)
            
            optimizer.zero_grad()
            
            pred_mal, pred_sev = model.forward_downstream(img_c, img_h)
            
            loss_mal = criterion_cls(pred_mal, target_mal)
            loss_sev = criterion_cls(pred_sev, target_sev)
            
            loss = loss_mal + loss_sev 
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically and at end
        if (epoch + 1) % config['logging_freq'] == 0 or (epoch + 1) == config['max_epochs']:
             if not os.path.exists(config['weights_save_path']):
                 os.makedirs(config['weights_save_path'])
             save_path = os.path.join(config['weights_save_path'], f"finetuned_epoch_{epoch+1}.pth")
             torch.save(model.state_dict(), save_path)
             print(f"Saved model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
