import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_encoder import DualEncoder
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_swede_scores(csv_path):
    """
    Parses swede_scores.csv.
    Assumes row index i corresponds to 'Case {i+1}'.
    Score < 5: Benign (0)
    Score >= 5: Malignant (1)
    """
    case_labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row: continue
            try:
                score = int(row[0])
                label = 1 if score >= 5 else 0
                case_name = f"Case {i+1}" # e.g., Case 1, Case 2...
                case_labels[case_name] = label
            except ValueError:
                continue
    return case_labels

class EvalDataset(Dataset):
    def __init__(self, root_dir, case_labels, transform=None):
        self.root_dir = root_dir
        self.case_labels = case_labels
        self.transform = transform
        self.items = []

        # Iterate through Case folders
        for case_name, label in case_labels.items():
            case_path = os.path.join(root_dir, case_name)
            if not os.path.exists(case_path):
                continue
            
            # Find images in case folder
            for fname in os.listdir(case_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(case_path, fname)
                    # We treat every image in the case as having that case's label
                    self.items.append((img_path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Cannot evaluate unreadable image: {img_path}") from e
        
        return image, label

def evaluate():
    config_path = 'configs/config.yaml'
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Data
    swede_path = os.path.join(config['dataset_path'], 'annocerv_real', 'swede_scores.csv')
    if not os.path.exists(swede_path):
        print(f"Error: {swede_path} not found.")
        return

    case_labels = parse_swede_scores(swede_path)
    print(f"Loaded labels for {len(case_labels)} cases.")

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])

    dataset = EvalDataset(os.path.join(config['dataset_path'], 'annocerv_real'), case_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    if len(dataset) == 0:
        raise ValueError("No evaluation images found; cannot report scores")
    print(f"Evaluating on {len(dataset)} images from AnnoCerv.")

    # Model
    model = DualEncoder(config).to(device)
    
    # Load Weights
    weights_path = os.path.join(config['weights_save_path'], 'finetuned_epoch_100.pth')
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Evaluation checkpoint not found: {weights_path}")
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True), strict=True)

    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            # For inference, if we don't have separate histo images, duplicate colpo
            # or update model to handle single input. 
            # Current DualEncoder expects (img_c, img_h).
            # We will pass same image to both for now as a fallback/proxy
            
            logits_mal, _ = model.forward_downstream(imgs, imgs)
            preds = torch.argmax(logits_mal, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_logits.extend(logits_mal.cpu().numpy())

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    report = classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'])
    print("\nClassification Report:")
    print(report)

    # Ensure viz dir
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # 1. Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    print("Saved visualizations/confusion_matrix.png")

    # 2. ROC Curve
    # softmax to get probs
    # logits_mal is [N, 2]
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    probs_mal = probs[:, 1] # Probability of Malignant
    
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    fpr, tpr, _ = roc_curve(all_labels, probs_mal)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('visualizations/roc_curve.png')
    plt.close()
    print(f"Saved visualizations/roc_curve.png (AUC: {roc_auc:.4f})")

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, probs_mal)
    avg_precision = average_precision_score(all_labels, probs_mal)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('visualizations/pr_curve.png')
    plt.close()
    print(f"Saved visualizations/pr_curve.png (AP: {avg_precision:.4f})")

if __name__ == "__main__":
    evaluate()
