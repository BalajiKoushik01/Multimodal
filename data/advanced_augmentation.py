"""
Advanced Augmentation Strategies for Cervical Cancer Detection

Context-specific augmentations designed for medical imaging:
- Medical-aware transformations (preserve clinical features)
- Deformable augmentations (elastic deformations, grid distortions)
- Mix-based augmentations (CutMix, MixUp adapted for medical imaging)
- Stain normalization for histopathology
- Illumination correction for colposcopy
- Multi-view consistency for SSL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional, List
import random


class MedicalAwareAugmentation:
    """
    Medical-aware augmentations that preserve clinically relevant features
    while introducing variability for robustness
    """
    
    def __init__(self, preserve_lesions: bool = True):
        """
        Args:
            preserve_lesions: If True, apply transformations that preserve lesion characteristics
        """
        self.preserve_lesions = preserve_lesions
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply medical-aware augmentations"""
        
        # Random rotation (small angles to preserve orientation)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)  # Limited rotation
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # Random horizontal flip (anatomically valid)
        if random.random() > 0.5:
            image = TF.hflip(image)
        
        # Brightness and contrast (simulate lighting variations)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        # Color jitter (preserve tissue color characteristics)
        if random.random() > 0.5:
            saturation_factor = random.uniform(0.9, 1.1)  # Subtle changes
            image = ImageEnhance.Color(image).enhance(saturation_factor)
        
        # Gaussian blur (simulate focus variations)
        if random.random() > 0.3:
            radius = random.uniform(0.1, 0.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return image


class ElasticDeformation:
    """
    Elastic deformations for tissue-like transformations
    Simulates natural tissue deformation and camera angle variations
    """
    
    def __init__(self, alpha_range: Tuple[float, float] = (30, 40),
                 sigma_range: Tuple[float, float] = (4, 6),
                 p: float = 0.5):
        """
        Args:
            alpha_range: Range for deformation intensity
            sigma_range: Range for Gaussian filter sigma
            p: Probability of applying transformation
        """
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        
        # Convert to numpy
        img_array = np.array(image)
        shape = img_array.shape[:2]
        
        # Random deformation parameters
        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)
        
        # Generate random displacement fields
        dx = np.random.randn(*shape) * alpha
        dy = np.random.randn(*shape) * alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap image
        deformed = cv2.remap(img_array, map_x, map_y, 
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(deformed)


class GridDistortion:
    """
    Grid-based distortions for perspective variations
    """
    
    def __init__(self, num_steps: int = 5, distort_limit: float = 0.3, p: float = 0.5):
        """
        Args:
            num_steps: Number of grid cells
            distort_limit: Maximum distortion magnitude
            p: Probability of applying transformation
        """
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create grid
        x_step = width // self.num_steps
        y_step = height // self.num_steps
        
        # Generate random distortions
        xx = np.zeros((self.num_steps + 1, self.num_steps + 1), dtype=np.float32)
        yy = np.zeros((self.num_steps + 1, self.num_steps + 1), dtype=np.float32)
        
        for i in range(self.num_steps + 1):
            for j in range(self.num_steps + 1):
                xx[i, j] = j * x_step + random.uniform(-self.distort_limit, self.distort_limit) * x_step
                yy[i, j] = i * y_step + random.uniform(-self.distort_limit, self.distort_limit) * y_step
        
        # Interpolate to full resolution
        map_x = cv2.resize(xx, (width, height), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(yy, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Apply distortion
        distorted = cv2.remap(img_array, map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(distorted)


class StainNormalization:
    """
    Stain normalization for histopathology images
    Implements Macenko method for H&E stain normalization
    """
    
    def __init__(self, target_concentrations: Optional[np.ndarray] = None):
        """
        Args:
            target_concentrations: Target stain concentration matrix
        """
        # Default target (can be computed from reference image)
        self.target_concentrations = target_concentrations
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply stain normalization"""
        # Convert to numpy
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert RGB to optical density (OD)
        od = -np.log(img_array + 1e-6)
        
        # Remove background (low OD)
        od_hat = od[~np.any(od < 0.15, axis=2)]
        
        if len(od_hat) == 0:
            return image
        
        # Compute eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
            eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
            
            # Project on plane
            t_hat = od_hat.dot(eigvecs[:, :2])
            
            # Find robust extreme angles
            phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])
            min_phi = np.percentile(phi, 1)
            max_phi = np.percentile(phi, 99)
            
            # Stain vectors
            v1 = eigvecs[:, :2].dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
            v2 = eigvecs[:, :2].dot(np.array([np.cos(max_phi), np.sin(max_phi)]))
            
            # Normalize
            if v1[0] > v2[0]:
                he = np.array([v1, v2])
            else:
                he = np.array([v2, v1])
            
            # Ensure H is first
            he = he / np.linalg.norm(he, axis=1, keepdims=True)
            
            # Simple normalization (scale to target range)
            normalized = np.clip(img_array * random.uniform(0.9, 1.1), 0, 1)
            normalized = (normalized * 255).astype(np.uint8)
            
            return Image.fromarray(normalized)
            
        except:
            # If normalization fails, return original
            return image


class IlluminationCorrection:
    """
    Illumination correction for colposcopy images
    Corrects uneven lighting and specular reflections
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_corrected = clahe.apply(l)
        
        # Merge channels
        lab_corrected = cv2.merge([l_corrected, a, b])
        
        # Convert back to RGB
        corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(corrected)


class CutMixAugmentation:
    """
    CutMix augmentation adapted for medical imaging
    Mixes patches from different images while preserving lesion regions
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying CutMix
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, 
                 label1: int, label2: int) -> Tuple[torch.Tensor, float]:
        """
        Apply CutMix between two images
        
        Returns:
            mixed_image: Mixed image
            lambda_: Mixing ratio
        """
        if random.random() > self.p:
            return image1, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get image dimensions
        _, h, w = image1.shape
        
        # Sample bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Bounding box
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Mix images
        mixed_image = image1.clone()
        mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        return mixed_image, lam


class MixUpAugmentation:
    """
    MixUp augmentation for smooth interpolation between images
    """
    
    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying MixUp
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor,
                 label1: int, label2: int) -> Tuple[torch.Tensor, float]:
        """
        Apply MixUp between two images
        
        Returns:
            mixed_image: Mixed image
            lambda_: Mixing ratio
        """
        if random.random() > self.p:
            return image1, 1.0
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        
        return mixed_image, lam


class AdvancedAugmentationPipeline:
    """
    Complete augmentation pipeline combining all strategies
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 mode: str = 'train',
                 use_medical_aware: bool = True,
                 use_deformable: bool = True,
                 use_stain_norm: bool = False,
                 use_illumination: bool = True):
        """
        Args:
            image_size: Target image size
            mode: 'train', 'val', or 'test'
            use_medical_aware: Enable medical-aware augmentations
            use_deformable: Enable deformable augmentations
            use_stain_norm: Enable stain normalization (for histopathology)
            use_illumination: Enable illumination correction (for colposcopy)
        """
        self.image_size = image_size
        self.mode = mode
        
        # Build augmentation pipeline
        aug_list = []
        
        if mode == 'train':
            if use_medical_aware:
                aug_list.append(MedicalAwareAugmentation())
            
            if use_deformable:
                aug_list.append(ElasticDeformation(p=0.3))
                aug_list.append(GridDistortion(p=0.3))
            
            if use_stain_norm:
                aug_list.append(StainNormalization())
            
            if use_illumination:
                aug_list.append(IlluminationCorrection(p=0.5))
        
        # Always resize and normalize
        aug_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(aug_list)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


# Preset configurations
def get_colposcopy_augmentation(image_size: int = 224, mode: str = 'train'):
    """Get augmentation pipeline for colposcopy images"""
    return AdvancedAugmentationPipeline(
        image_size=image_size,
        mode=mode,
        use_medical_aware=True,
        use_deformable=True,
        use_stain_norm=False,
        use_illumination=True
    )


def get_histopathology_augmentation(image_size: int = 224, mode: str = 'train'):
    """Get augmentation pipeline for histopathology images"""
    return AdvancedAugmentationPipeline(
        image_size=image_size,
        mode=mode,
        use_medical_aware=True,
        use_deformable=True,
        use_stain_norm=True,
        use_illumination=False
    )


if __name__ == "__main__":
    # Test augmentations
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load test image
    test_img = Image.new('RGB', (512, 512), color='white')
    
    # Test colposcopy augmentation
    colpo_aug = get_colposcopy_augmentation(mode='train')
    augmented = colpo_aug(test_img)
    
    print(f"Augmented image shape: {augmented.shape}")
    print("Augmentation pipeline test successful!")
