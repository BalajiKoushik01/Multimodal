import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import yaml
import csv

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_encoder import DualEncoder

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_swede_scores(csv_path):
    case_labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row: continue
            try:
                score = int(row[0])
                label = 1 if score >= 5 else 0
                case_name = f"Case {i+1}"
                case_labels[case_name] = label
            except ValueError:
                continue
    return case_labels

def get_image_paths(root_dir, case_labels):
    images = []
    labels = []
    # AnnoCerv
    for case_name, label in case_labels.items():
        case_path = os.path.join(root_dir, 'annocerv_real', case_name)
        if os.path.exists(case_path):
            for fname in os.listdir(case_path):
                if fname.lower().endswith(('.jpg', '.png')):
                    images.append(os.path.join(case_path, fname))
                    labels.append(label) # 0 or 1
    return images, labels

def visualize():
    config_path = 'configs/config.yaml'
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = DualEncoder(config).to(device)
    weights_path = os.path.join(config['weights_save_path'], 'finetuned_epoch_100.pth')
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    else:
        print("Weights not found.")
        return
    model.eval()

    # 2. Prepare Data
    swede_path = os.path.join(config['dataset_path'], 'annocerv_real', 'swede_scores.csv')
    if not os.path.exists(swede_path):
        print("Swede scores not found.")
        return

    case_labels = parse_swede_scores(swede_path)
    image_paths, labels = get_image_paths(config['dataset_path'], case_labels)
    
    # Limit to e.g. 500 images for speed if too many
    if len(image_paths) > 1000:
        indices = np.random.choice(len(image_paths), 1000, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"Extracting features for {len(image_paths)} images...")

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])

    features = []
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                
                # Get backbone features or projection head features
                # DualEncoder forward_ssl returns z_c, z_h (projected)
                # We use z_c (colposcopy projection)
                z_c, _, _, _ = model.forward_ssl(img_t, img_t)
                features.append(z_c.cpu().numpy().flatten())
                
                if (i+1) % 50 == 0:
                    print(f"Processed {i+1}/{len(image_paths)}")
            except Exception as e:
                print(f"Error {e}")

    features = np.array(features)
    labels = np.array(labels)

    # 3. Runs t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # 4. Plot
    plt.figure(figsize=(10, 8))
    
    # Benign
    idx_0 = labels == 0
    plt.scatter(features_2d[idx_0, 0], features_2d[idx_0, 1], c='blue', label='Benign', alpha=0.6, s=10)
    
    # Malignant
    idx_1 = labels == 1
    plt.scatter(features_2d[idx_1, 0], features_2d[idx_1, 1], c='red', label='Malignant', alpha=0.6, s=10)
    
    plt.title('t-SNE of Colposcopy Image Embeddings')
    plt.legend()
    plt.grid(True)
    
    save_path = 'visualizations/tsne_embeddings.png'
    plt.savefig(save_path)
    print(f"Saved t-SNE plot to {save_path}")

if __name__ == "__main__":
    visualize()
