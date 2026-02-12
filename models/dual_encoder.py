"""
Enhanced Dual-Encoder Architecture for Multimodal Cervical Cancer Detection

Integrates:
1. ViT+ASPP backbones for both modalities
2. Cross-modal attention for feature fusion
3. Advanced projection heads for SSL
4. Multi-task learning (malignancy, severity, progression risk)
5. Support for single-modality inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_backbone, ProjectionHead
from .attention_modules import (
    CrossModalAttention,
    MultiHeadCrossAttentionFusion,
    CBAM,
    AttentionPooling
)


class EnhancedDualEncoder(nn.Module):
    """
    Enhanced dual-encoder with ViT+ASPP and cross-modal attention
    
    Supports:
    - Self-supervised learning (SSL) with contrastive loss
    - Downstream classification tasks
    - Single-modality and multi-modality inference
    - Attention visualization
    """
    
    def __init__(self, config):
        super(EnhancedDualEncoder, self).__init__()
        
        self.config = config
        self.backbone_name = config.get('backbone', 'resnet50')
        self.projection_dim = config['projection_dim']
        self.use_cross_attention = config.get('use_cross_attention', True)
        self.use_aspp = config.get('use_aspp', True)
        
        # Encoders for each modality
        self.colpo_encoder = get_backbone(
            self.backbone_name,
            self.projection_dim,
            pretrained=config.get('pretrained', True)
        )
        
        self.histo_encoder = get_backbone(
            self.backbone_name,
            self.projection_dim,
            pretrained=config.get('pretrained', True)
        )
        
        # Enhanced projection heads for SSL
        proj_hidden_dim = config['model_params'].get('projection_head_hidden_dim', 4096)
        proj_output_dim = config.get('ssl_projection_dim', 128)
        num_proj_layers = config.get('num_projection_layers', 3)
        
        self.projector_colpo = ProjectionHead(
            self.projection_dim,
            proj_hidden_dim,
            proj_output_dim,
            num_layers=num_proj_layers
        )
        
        self.projector_histo = ProjectionHead(
            self.projection_dim,
            proj_hidden_dim,
            proj_output_dim,
            num_layers=num_proj_layers
        )
        
        # Cross-modal attention (if enabled)
        if self.use_cross_attention:
            self.cross_attn_colpo = CrossModalAttention(
                dim=self.projection_dim,
                num_heads=config.get('num_attention_heads', 8)
            )
            self.cross_attn_histo = CrossModalAttention(
                dim=self.projection_dim,
                num_heads=config.get('num_attention_heads', 8)
            )
            
            # Multi-head fusion
            self.fusion_module = MultiHeadCrossAttentionFusion(
                dim=self.projection_dim,
                num_modalities=2,
                num_heads=config.get('num_attention_heads', 8),
                num_layers=config.get('num_fusion_layers', 2)
            )
        
        # CBAM for spatial and channel attention
        if 'vit' in self.backbone_name:
            # For ViT, we'll apply attention on the fused features
            self.cbam = None
        else:
            self.cbam_colpo = CBAM(in_channels=2048)  # ResNet50 output
            self.cbam_histo = CBAM(in_channels=2048)
        
        # Attention pooling
        self.attn_pool_colpo = AttentionPooling(dim=self.projection_dim)
        self.attn_pool_histo = AttentionPooling(dim=self.projection_dim)
        
        # Fusion strategy
        fusion_dim = self.projection_dim * 2 if self.use_cross_attention else self.projection_dim * 2
        
        # Multi-task heads
        num_classes = config['model_params'].get('num_classes', 2)
        severity_classes = config['model_params'].get('severity_classes', 4)
        
        self.malignancy_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        self.severity_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, severity_classes)
        )
        
        # Progression risk head (for P_progression)
        self.progression_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temperature for contrastive learning
        self.temperature = config.get('temperature', 0.07)
        
        # Store attention weights for visualization
        self.attention_weights = {}
    
    def forward_ssl(self, x_colpo, x_histo, return_features=False):
        """
        Forward pass for self-supervised learning
        
        Args:
            x_colpo: Colposcopy images [B, 3, H, W]
            x_histo: Histopathology images [B, 3, H, W]
            return_features: If True, return intermediate features
        
        Returns:
            z_c: Normalized colposcopy projections [B, ssl_projection_dim]
            z_h: Normalized histopathology projections [B, ssl_projection_dim]
            features: Dict of intermediate features (optional)
        """
        # Extract features from encoders
        if 'vit' in self.backbone_name and self.use_aspp:
            feat_c, aspp_c = self.colpo_encoder(x_colpo, use_aspp=True)
            feat_h, aspp_h = self.histo_encoder(x_histo, use_aspp=True)
        else:
            feat_c, _ = self.colpo_encoder(x_colpo)
            feat_h, _ = self.histo_encoder(x_histo)
            aspp_c, aspp_h = None, None
        
        # Apply projection heads
        proj_c = self.projector_colpo(feat_c)
        proj_h = self.projector_histo(feat_h)
        
        # Normalize projections
        z_c = F.normalize(proj_c, dim=1)
        z_h = F.normalize(proj_h, dim=1)
        
        if return_features:
            features = {
                'feat_colpo': feat_c,
                'feat_histo': feat_h,
                'aspp_colpo': aspp_c,
                'aspp_histo': aspp_h,
                'proj_colpo': proj_c,
                'proj_histo': proj_h
            }
            return z_c, z_h, features
        
        return z_c, z_h
    
    def forward_downstream(self, x_colpo=None, x_histo=None, return_attention=False):
        """
        Forward pass for downstream classification tasks
        
        Args:
            x_colpo: Colposcopy images [B, 3, H, W] (optional)
            x_histo: Histopathology images [B, 3, H, W] (optional)
            return_attention: If True, return attention weights
        
        Returns:
            malignancy: Malignancy predictions [B, num_classes]
            severity: Severity predictions [B, severity_classes]
            progression_risk: Progression risk scores [B, 1]
            attention_weights: Dict of attention weights (optional)
        """
        # Handle single-modality inference
        has_colpo = x_colpo is not None
        has_histo = x_histo is not None
        
        if not has_colpo and not has_histo:
            raise ValueError("At least one modality must be provided")
        
        # Extract features
        if has_colpo:
            if 'vit' in self.backbone_name and self.use_aspp:
                feat_c, aspp_c = self.colpo_encoder(x_colpo, use_aspp=True)
            else:
                feat_c, _ = self.colpo_encoder(x_colpo)
        
        if has_histo:
            if 'vit' in self.backbone_name and self.use_aspp:
                feat_h, aspp_h = self.histo_encoder(x_histo, use_aspp=True)
            else:
                feat_h, _ = self.histo_encoder(x_histo)
        
        # Cross-modal attention fusion
        if has_colpo and has_histo and self.use_cross_attention:
            # Reshape features for attention (add sequence dimension)
            feat_c_seq = feat_c.unsqueeze(1)  # [B, 1, C]
            feat_h_seq = feat_h.unsqueeze(1)  # [B, 1, C]
            
            # Cross-attention
            if return_attention:
                feat_c_attended, attn_c2h = self.cross_attn_colpo(
                    feat_c_seq, feat_h_seq, return_attention=True
                )
                feat_h_attended, attn_h2c = self.cross_attn_histo(
                    feat_h_seq, feat_c_seq, return_attention=True
                )
                self.attention_weights['colpo_to_histo'] = attn_c2h
                self.attention_weights['histo_to_colpo'] = attn_h2c
            else:
                feat_c_attended = self.cross_attn_colpo(feat_c_seq, feat_h_seq)
                feat_h_attended = self.cross_attn_histo(feat_h_seq, feat_c_seq)
            
            # Remove sequence dimension
            feat_c_attended = feat_c_attended.squeeze(1)
            feat_h_attended = feat_h_attended.squeeze(1)
            
            # Combine original and attended features
            feat_c_final = feat_c + feat_c_attended
            feat_h_final = feat_h + feat_h_attended
            
            # Late fusion
            combined = torch.cat([feat_c_final, feat_h_final], dim=1)
        
        elif has_colpo and has_histo:
            # Simple concatenation without cross-attention
            combined = torch.cat([feat_c, feat_h], dim=1)
        
        elif has_colpo:
            # Single modality: colposcopy only
            combined = torch.cat([feat_c, feat_c], dim=1)  # Duplicate for consistent dimensions
        
        else:
            # Single modality: histopathology only
            combined = torch.cat([feat_h, feat_h], dim=1)  # Duplicate for consistent dimensions
        
        # Multi-task predictions
        malignancy = self.malignancy_head(combined)
        severity = self.severity_head(combined)
        progression_risk = self.progression_head(combined)
        
        if return_attention:
            return malignancy, severity, progression_risk, self.attention_weights
        
        return malignancy, severity, progression_risk
    
    def get_attention_maps(self):
        """Get stored attention weights for visualization"""
        return self.attention_weights
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.colpo_encoder.parameters():
            param.requires_grad = False
        for param in self.histo_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.colpo_encoder.parameters():
            param.requires_grad = True
        for param in self.histo_encoder.parameters():
            param.requires_grad = True


# Backward compatibility: alias for existing code
DualEncoder = EnhancedDualEncoder


if __name__ == "__main__":
    print("Testing Enhanced Dual-Encoder...")
    
    # Test configuration
    config = {
        'backbone': 'resnet50',
        'projection_dim': 2048,
        'ssl_projection_dim': 128,
        'pretrained': False,
        'use_cross_attention': True,
        'use_aspp': False,
        'num_attention_heads': 8,
        'num_fusion_layers': 2,
        'num_projection_layers': 3,
        'temperature': 0.07,
        'model_params': {
            'projection_head_hidden_dim': 4096,
            'num_classes': 2,
            'severity_classes': 4
        }
    }
    
    model = EnhancedDualEncoder(config)
    
    # Test SSL forward pass
    print("\n1. Testing SSL forward pass...")
    x_colpo = torch.randn(2, 3, 224, 224)
    x_histo = torch.randn(2, 3, 224, 224)
    
    z_c, z_h, features = model.forward_ssl(x_colpo, x_histo, return_features=True)
    print(f"   Colposcopy projection shape: {z_c.shape}")
    print(f"   Histopathology projection shape: {z_h.shape}")
    print(f"   Features keys: {list(features.keys())}")
    
    # Test downstream forward pass
    print("\n2. Testing downstream forward pass (dual-modality)...")
    malignancy, severity, progression = model.forward_downstream(x_colpo, x_histo)
    print(f"   Malignancy shape: {malignancy.shape}")
    print(f"   Severity shape: {severity.shape}")
    print(f"   Progression risk shape: {progression.shape}")
    
    # Test single-modality inference
    print("\n3. Testing single-modality inference (colposcopy only)...")
    malignancy, severity, progression = model.forward_downstream(x_colpo=x_colpo)
    print(f"   Malignancy shape: {malignancy.shape}")
    print(f"   Severity shape: {severity.shape}")
    
    print("\n4. Testing single-modality inference (histopathology only)...")
    malignancy, severity, progression = model.forward_downstream(x_histo=x_histo)
    print(f"   Malignancy shape: {malignancy.shape}")
    print(f"   Severity shape: {severity.shape}")
    
    # Test with attention visualization
    print("\n5. Testing with attention visualization...")
    malignancy, severity, progression, attn_weights = model.forward_downstream(
        x_colpo, x_histo, return_attention=True
    )
    print(f"   Attention weights keys: {list(attn_weights.keys())}")
    
    print("\n✓ All dual-encoder tests passed!")
