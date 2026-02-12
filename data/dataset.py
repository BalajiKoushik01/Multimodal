import os
import torch
import csv
import re
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys
# Add parent dir for imports if run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.external_data import parse_annocerv, parse_mobileodt

class CervicalDataset(Dataset):
    def __init__(self, root_dir, modality='colposcopy', transform=None, mode='train'):
        """
        Args:
            root_dir (str): Directory with all the images.
            modality (str): 'colposcopy' or 'histopathology'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.root_dir = root_dir
        self.modality = modality
        self.transform = transform
        self.mode = mode
        
        # Assume images are stored in subfolders by class if labeled, 
        # or flat folder if unlabeled/ssl
        # For simplicity, let's assume a flat folder for SSL pretraining
        
        self.image_paths = []
        self.dataset_type = 'folder' # Default
        
        # Check if user specified a dataset type or infer from structure
        self.labels_map = {}
        
        # If modality is 'annocerv', try to parse annocerv structure
        if modality.lower() == 'annocerv':
             self.dataset_type = 'annocerv'
             self.items = parse_annocerv(root_dir) # List of dicts
             
             # Load labels from swede_scores.csv if available
             swede_path = os.path.join(root_dir, 'swede_scores.csv')
             if os.path.exists(swede_path):
                 print(f"Loading labels from {swede_path}")
                 with open(swede_path, 'r') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if not row: continue
                        try:
                            score = int(row[0])
                            label = 1 if score >= 5 else 0
                            # Map directory name "Case 1", "Case 2" etc. to label
                            self.labels_map[f"Case {i+1}"] = label
                        except ValueError:
                            pass
                            
        elif modality.lower() == 'mobileodt':
             self.dataset_type = 'mobileodt'
             self.items = parse_mobileodt(root_dir, mode)
        else:
             # Standard behavior
             target_dir = os.path.join(root_dir, modality)
             if os.path.exists(target_dir):
                  self.items = [{'image_path': os.path.join(target_dir, f)} for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
             else:
                 print(f"Warning: Directory {target_dir} not found.")
                 self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_name = item['image_path']
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Determine label
        label = -1
        if self.dataset_type == 'annocerv':
            # Extract Case Name from path (e.g., .../Case 1/img.jpg)
            # Assuming structure: root/Case X/img.jpg
            path_parts = os.path.normpath(img_name).split(os.sep)
            # Case folder should be the parent
            parent_dir = path_parts[-2]
            if parent_dir in self.labels_map:
                label = self.labels_map[parent_dir]
        elif self.dataset_type == 'mobileodt':
            # parse_mobileodt returns 'class_idx' in the item dict
            # 0: Type_1 (Benign), 1: Type_2 (Atypical), 2: Type_3 (Malignant)
            # Map to Binary (0: Benign, 1: Malignant)?
            # Strategy: Type_1 (0) -> 0, Type_2 (1) -> 0 (or 1?), Type_3 (2) -> 1.
            # Safe bet: Type_1=0, Type_3=1. Type_2 maybe ignore or map to 1 (pre-cancer).
            # Let's map Type_1 -> 0, Type_2 -> 1, Type_3 -> 1 (Risk vs No Risk)
            source_idx = item.get('class_idx', -1)
            if source_idx == 0: # Type 1
                label = 0
            elif source_idx == 2: # Type 3
                label = 1
            elif source_idx == 1: # Type 2
                label = 1 # Treat atypical as "positive" for screening
            else:
                label = -1
        
        # In SSL, we might return two views of the same image
        # For cross-domain, we might return unpaired samples from both domains
        # This implementation returns a single image for now
        
        return image, label

class LesionAwareDataset(Dataset):
    """
    Dataset wrapper that applies lesion-aware augmentations.
    """
    def __init__(self, base_dataset, lesion_crop_prob=0.5):
        self.base_dataset = base_dataset
        self.lesion_crop_prob = lesion_crop_prob

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Implementation of lesion-aware logic would go here
        # For now, just delegate to base dataset
        return self.base_dataset[idx]

class CombinedDataset(Dataset):
    """
    Dataset that combines multiple sources (mock, annocerv, mobileodt).
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)
        self.cumulative_lengths = [0] + [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_len:
                dataset_idx = i
                internal_idx = idx - self.cumulative_lengths[i]
                return self.datasets[dataset_idx][internal_idx]
        
        raise IndexError(f"Index {idx} out of range")
