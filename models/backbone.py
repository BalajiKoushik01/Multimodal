"""
Enhanced Backbone Architectures for Cervical Cancer Detection

Implements:
1. ResNet50 backbone (baseline)
2. Vision Transformer (ViT) with ASPP for multi-scale feature extraction
3. Hybrid CNN-ViT architecture
4. Projection heads for self-supervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. ViT backbones will not be available.")


class ResNetBackbone(nn.Module):
    """ResNet50 backbone for baseline comparison"""
    
    def __init__(self, output_dim, pretrained=True):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc layer
        self.output_dim = output_dim
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        projection = self.fc(features)
        return features, projection


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    
    Multi-scale feature extraction using parallel atrous convolutions
    with different dilation rates. Critical for precise lesion boundary
    delineation in CIN grading.
    """
    
    def __init__(self, in_channels, out_channels=256, dilations=[6, 12, 18, 24]):
        super(ASPPModule, self).__init__()
        
        self.dilations = dilations
        
        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolution branches
        self.atrous_convs = nn.ModuleList()
        for dilation in dilations:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        num_branches = 2 + len(dilations)  # 1x1 + atrous + global
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        # 1x1 convolution
        out1 = self.conv1x1(x)
        
        # Atrous convolutions
        atrous_outs = [conv(x) for conv in self.atrous_convs]
        
        # Global average pooling
        global_out = self.global_avg_pool(x)
        global_out = F.interpolate(global_out, size=x.shape[2:], 
                                   mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        out = torch.cat([out1] + atrous_outs + [global_out], dim=1)
        
        # Fuse
        out = self.fusion(out)
        
        return out


class ViTASPPBackbone(nn.Module):
    """
    Vision Transformer with ASPP for multi-scale feature extraction
    
    Combines global attention from ViT with multi-scale context from ASPP.
    Ideal for capturing both fine-grained cellular details and global tissue architecture.
    """
    
    def __init__(self, output_dim, model_name='vit_base_patch16_224', 
                 pretrained=True, aspp_dilations=[6, 12, 18, 24]):
        super(ViTASPPBackbone, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT backbones. Install with: pip install timm")
        
        # Load pre-trained ViT
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get ViT feature dimension
        if 'base' in model_name:
            vit_dim = 768
            patch_size = 16
        elif 'large' in model_name:
            vit_dim = 1024
            patch_size = 16
        elif 'small' in model_name:
            vit_dim = 384
            patch_size = 16
        else:
            vit_dim = 768  # Default
            patch_size = 16
        
        self.vit_dim = vit_dim
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # ASPP module for multi-scale features
        self.aspp = ASPPModule(in_channels=vit_dim, out_channels=256, 
                              dilations=aspp_dilations)
        
        # Projection to output dimension
        self.fc = nn.Linear(256, output_dim)
        
        # For global features (alternative path without ASPP)
        self.global_fc = nn.Linear(vit_dim, output_dim)
    
    def forward(self, x, use_aspp=True):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            use_aspp: If True, use ASPP for multi-scale features. 
                     If False, use global pooling only.
        
        Returns:
            features: Feature tensor [B, output_dim]
            aspp_features: ASPP feature maps [B, 256, H', W'] (if use_aspp=True)
        """
        B = x.shape[0]
        
        # ViT feature extraction
        vit_features = self.vit.forward_features(x)  # [B, num_patches+1, vit_dim]
        
        # Separate CLS token and patch tokens
        cls_token = vit_features[:, 0, :]  # [B, vit_dim]
        patch_tokens = vit_features[:, 1:, :]  # [B, num_patches, vit_dim]
        
        if use_aspp:
            # Reshape patch tokens to spatial format for ASPP
            # Assuming 224x224 input with patch_size=16 -> 14x14 patches
            num_patches = patch_tokens.shape[1]
            h = w = int(num_patches ** 0.5)
            
            spatial_features = patch_tokens.transpose(1, 2).reshape(B, self.vit_dim, h, w)
            
            # Apply ASPP
            aspp_features = self.aspp(spatial_features)  # [B, 256, h, w]
            
            # Global average pooling
            global_features = F.adaptive_avg_pool2d(aspp_features, 1).flatten(1)  # [B, 256]
            
            # Project to output dimension
            features = self.fc(global_features)  # [B, output_dim]
            
            return features, aspp_features
        else:
            # Use CLS token directly
            features = self.global_fc(cls_token)  # [B, output_dim]
            return features, None


class HybridCNNViTBackbone(nn.Module):
    """
    Hybrid CNN-ViT backbone with ASPP
    
    Combines local feature extraction from CNN with global context from ViT.
    """
    
    def __init__(self, output_dim, vit_model_name='vit_base_patch16_224', 
                 pretrained=True):
        super(HybridCNNViTBackbone, self).__init__()
        
        # CNN stem (ResNet first 3 blocks)
        resnet = models.resnet50(pretrained=pretrained)
        self.cnn_stem = nn.Sequential(*list(resnet.children())[:6])  # Up to layer2
        
        # ViT for global features
        if TIMM_AVAILABLE:
            self.vit = timm.create_model(vit_model_name, pretrained=pretrained, 
                                        num_classes=0, img_size=56)  # Smaller input after CNN
            vit_dim = 768
        else:
            raise ImportError("timm is required for hybrid backbone")
        
        # ASPP on CNN features
        self.aspp = ASPPModule(in_channels=512, out_channels=256)  # ResNet layer2 has 512 channels
        
        # Fusion layer
        self.fusion = nn.Linear(256 + vit_dim, output_dim)
        self.output_dim = output_dim
    
    def forward(self, x):
        # CNN features
        cnn_features = self.cnn_stem(x)  # [B, 512, 56, 56]
        
        # ASPP on CNN features
        aspp_features = self.aspp(cnn_features)  # [B, 256, 56, 56]
        aspp_global = F.adaptive_avg_pool2d(aspp_features, 1).flatten(1)  # [B, 256]
        
        # ViT features (on CNN output)
        vit_features = self.vit.forward_features(cnn_features)[:, 0, :]  # [B, vit_dim]
        
        # Fuse CNN and ViT features
        combined = torch.cat([aspp_global, vit_features], dim=1)
        features = self.fusion(combined)
        
        return features, aspp_features


def get_backbone(name, output_dim, pretrained=True, **kwargs):
    """
    Factory function to get backbone by name
    
    Args:
        name: Backbone name ('resnet50', 'vit_base', 'vit_large', 'hybrid')
        output_dim: Output feature dimension
        pretrained: Use pre-trained weights
        **kwargs: Additional arguments for specific backbones
    
    Returns:
        Backbone model
    """
    if name == 'resnet50':
        return ResNetBackbone(output_dim, pretrained)
    
    elif name == 'vit_base':
        return ViTASPPBackbone(output_dim, model_name='vit_base_patch16_224', 
                              pretrained=pretrained, **kwargs)
    
    elif name == 'vit_large':
        return ViTASPPBackbone(output_dim, model_name='vit_large_patch16_224', 
                              pretrained=pretrained, **kwargs)
    
    elif name == 'vit_small':
        return ViTASPPBackbone(output_dim, model_name='vit_small_patch16_224', 
                              pretrained=pretrained, **kwargs)
    
    elif name == 'hybrid':
        return HybridCNNViTBackbone(output_dim, pretrained=pretrained, **kwargs)
    
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose from: resnet50, vit_base, vit_large, vit_small, hybrid")


class ProjectionHead(nn.Module):
    """
    Non-linear projection head for self-supervised learning
    
    Enhanced version with batch normalization and deeper architecture
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(ProjectionHead, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    # Test backbones
    print("Testing backbone architectures...")
    
    # Test ResNet50
    print("\n1. Testing ResNet50 backbone...")
    resnet_backbone = get_backbone('resnet50', output_dim=2048, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features, projection = resnet_backbone(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Features shape: {features.shape}")
    print(f"   Projection shape: {projection.shape}")
    
    # Test ViT+ASPP (if timm available)
    if TIMM_AVAILABLE:
        print("\n2. Testing ViT+ASPP backbone...")
        vit_backbone = get_backbone('vit_base', output_dim=2048, pretrained=False)
        features, aspp_features = vit_backbone(x, use_aspp=True)
        print(f"   Input shape: {x.shape}")
        print(f"   Features shape: {features.shape}")
        print(f"   ASPP features shape: {aspp_features.shape}")
        
        print("\n3. Testing Hybrid CNN-ViT backbone...")
        hybrid_backbone = get_backbone('hybrid', output_dim=2048, pretrained=False)
        features, aspp_features = hybrid_backbone(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Features shape: {features.shape}")
        print(f"   ASPP features shape: {aspp_features.shape}")
    else:
        print("\n2-3. Skipping ViT tests (timm not available)")
    
    # Test projection head
    print("\n4. Testing projection head...")
    proj_head = ProjectionHead(input_dim=2048, hidden_dim=4096, output_dim=128, num_layers=3)
    proj_out = proj_head(features)
    print(f"   Input shape: {features.shape}")
    print(f"   Output shape: {proj_out.shape}")
    
    print("\n✓ All backbone tests passed!")
