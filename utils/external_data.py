
import os
import glob
import numpy as np
from PIL import Image

def parse_annocerv(root_dir):
    """
    Parses the AnnoCerv dataset structure.
    Returns a list of dictionaries, each containing paths to images and masks for a case.
    
    Structure:
    root_dir/
      Case 1/
        C1Aceto (1).jpg
        C1Aceto (1).png (mask)
        C1Iodine (1).jpg
        ...
    """
    cases = []
    # Case folders usually named "Case X" or similar
    # We'll just look for subdirectories
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for case_name in subdirs:
        case_path = os.path.join(root_dir, case_name)
        
        # Find all jpg images
        images = glob.glob(os.path.join(case_path, "*.jpg"))
        
        for img_path in images:
            # Check for corresponding png mask
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(case_path, base_name + ".png")
            
            item = {
                'image_path': img_path,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'case_id': case_name,
                'type': 'Aceto' if 'Aceto' in base_name else ('Iodine' if 'Iodine' in base_name else 'Green'),
                'dataset': 'AnnoCerv'
            }
            cases.append(item)
            
    return cases

def parse_mobileodt(root_dir, split='train'):
    """
    Parses the Intel MobileODT Kaggle dataset structure.
    
    Structure:
    root_dir/
      train/
        Type_1/
        Type_2/
        Type_3/
      test/
    """
    items = []
    
    data_path = os.path.join(root_dir, split)
    if not os.path.exists(data_path):
        # Fallback if user points directly to 'train' folder
        if os.path.exists(os.path.join(root_dir, 'Type_1')):
            data_path = root_dir
        else:
            print(f"Warning: MobileODT {split} folder not found in {root_dir}")
            return []

    # Classes: Type_1, Type_2, Type_3
    # Additional folder: 'additional_Type_1_v2', etc. (if using the full dataset)
    
    classes = ['Type_1', 'Type_2', 'Type_3']
    
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        if not os.path.exists(cls_path):
            continue
            
        images = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))
        
        for img_path in images:
            items.append({
                'image_path': img_path,
                'label': cls,
                'class_idx': classes.index(cls), # 0, 1, 2
                'dataset': 'MobileODT'
            })
            
    return items

def decode_annocerv_mask(mask):
    """
    Decodes the color-coded AnnoCerv mask into a semantic map.
    
    Colors:
    - Blue: Squamous-cylindrical junction
    - Purple: Aceto-white areas
    - Red: Atypical vessels and punctations
    - Brown: Mosaics
    - Yellow: Naboth cysts
    - Black: Cuffed gland openings
    """
    # Define colors (R, G, B) - need to verify exact values, usually pure colors
    # Assuming standard simplified colors for now
    # Open mask, convert to RGB
    mask_rgb = np.array(mask.convert('RGB'))
    h, w, _ = mask_rgb.shape
    
    semantic_map = np.zeros((h, w), dtype=np.uint8)
    
    # Simple thresholding/color matching
    # This is an approximation. Real values should be checked with sample data.
    
    # Purple (Aceto-white) -> Label 1 (Lesion)
    # Red (Vessels) -> Label 2
    # Brown (Mosaics) -> Label 3
    # ...
    
    # Example logic:
    # If pixel is Purple (R>100, G<100, B>100) -> 1
    purple_mask = (mask_rgb[:, :, 0] > 100) & (mask_rgb[:, :, 1] < 100) & (mask_rgb[:, :, 2] > 100)
    semantic_map[purple_mask] = 1
    
    return Image.fromarray(semantic_map)
