
import os
import argparse
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import yaml

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_encoder import DualEncoder

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def inference(image_path_colpo, image_path_histo, model_path, config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(config_path)
    
    # Model
    # A missing or incompatible checkpoint must never produce medical-looking predictions.
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found: {model_path}")
    model = DualEncoder(config).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    
    model.eval()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])

    # Load images
    try:
        img_c = Image.open(image_path_colpo).convert('RGB')
        img_h = Image.open(image_path_histo).convert('RGB')
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    input_c = transform(img_c).unsqueeze(0).to(device)
    input_h = transform(img_h).unsqueeze(0).to(device)

    with torch.no_grad():
        mal_logits, sev_logits = model.forward_downstream(input_c, input_h)
        mal_prob = F.softmax(mal_logits, dim=1)
        sev_prob = F.softmax(sev_logits, dim=1)

    mal_pred = torch.argmax(mal_prob, dim=1).item()
    sev_pred = torch.argmax(sev_prob, dim=1).item()

    classes = ["Benign", "Malignant"]
    severity = ["Low", "Moderate", "Severe"]

    print(f"Prediction for:\n  Colposcopy: {image_path_colpo}\n  Histopathology: {image_path_histo}")
    print(f"  Malignancy: {classes[mal_pred]} (Confidence: {mal_prob[0][mal_pred]:.2f})")
    print(f"  Severity: {severity[sev_pred]} (Confidence: {sev_prob[0][sev_pred]:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--colpo', type=str, required=True, help='Path to colposcopy image')
    parser.add_argument('--histo', type=str, required=True, help='Path to histopathology image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='Path to model weights')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    inference(args.colpo, args.histo, args.model, args.config)
