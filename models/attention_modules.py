"""
Advanced Attention Mechanisms for Multimodal Fusion

Implements:
1. Cross-modal attention between colposcopy and histopathology
2. Self-attention refinement for feature enhancement
3. Spatial attention for lesion localization
4. Channel attention for discriminative features
5. Attention visualization for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism
    
    Allows one modality to attend to features from another modality.
    Critical for integrating complementary information from colposcopy and histopathology.
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Query from modality 1, Key and Value from modality 2
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x1, x2, return_attention=False):
        """
        Args:
            x1: Query features [B, N1, C]
            x2: Key-Value features [B, N2, C]
            return_attention: If True, return attention weights
        
        Returns:
            out: Attended features [B, N1, C]
            attn_weights: Attention weights [B, num_heads, N1, N2] (optional)
        """
        B, N1, C = x1.shape
        N2 = x2.shape[1]
        
        # Query from x1
        q = self.q(x1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Key and Value from x2
        kv = self.kv(x2).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class SelfAttentionRefinement(nn.Module):
    """
    Self-attention refinement module
    
    Refines features using self-attention to capture long-range dependencies.
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super(SelfAttentionRefinement, self).__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input features [B, N, C]
            return_attention: If True, return attention weights
        
        Returns:
            out: Refined features [B, N, C]
            attn_weights: Attention weights [B, num_heads, N, N] (optional)
        """
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        # Residual connection
        out = x + x_attn
        out = self.norm(out)
        
        if return_attention:
            return out, attn
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module
    
    Focuses on important spatial regions (e.g., lesion areas).
    """
    
    def __init__(self, in_channels, reduction=16):
        super(SpatialAttention, self).__init__()
        
        # Channel pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Spatial attention
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            out: Spatially attended features [B, C, H, W]
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        pool_out = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.conv(pool_out)
        
        # Apply attention
        out = x * attn_map
        
        return out, attn_map


class ChannelAttention(nn.Module):
    """
    Channel attention module (Squeeze-and-Excitation)
    
    Emphasizes discriminative feature channels.
    """
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            out: Channel-attended features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Global pooling
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        
        # Channel attention
        avg_attn = self.fc(avg_out).view(B, C, 1, 1)
        max_attn = self.fc(max_out).view(B, C, 1, 1)
        
        attn = avg_attn + max_attn
        
        # Apply attention
        out = x * attn
        
        return out, attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Combines channel and spatial attention sequentially.
    """
    
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(in_channels)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            out: Attended features [B, C, H, W]
            channel_attn: Channel attention weights
            spatial_attn: Spatial attention map
        """
        # Channel attention
        x, channel_attn = self.channel_attn(x)
        
        # Spatial attention
        x, spatial_attn = self.spatial_attn(x)
        
        return x, channel_attn, spatial_attn


class MultiHeadCrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention fusion for multiple modalities
    
    Fuses features from colposcopy, histopathology, and other modalities
    using multi-head cross-attention.
    """
    
    def __init__(self, dim, num_modalities=2, num_heads=8, num_layers=2):
        super(MultiHeadCrossAttentionFusion, self).__init__()
        
        self.num_modalities = num_modalities
        self.num_layers = num_layers
        
        # Cross-attention layers for each modality pair
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_attns = nn.ModuleList()
            for _ in range(num_modalities):
                layer_attns.append(CrossModalAttention(dim, num_heads))
            self.cross_attn_layers.append(layer_attns)
        
        # Feed-forward networks
        self.ffns = nn.ModuleList()
        for _ in range(num_layers):
            self.ffns.append(nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim),
                nn.Dropout(0.1)
            ))
        
        # Layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers * 2)])
    
    def forward(self, modality_features, return_attention=False):
        """
        Args:
            modality_features: List of feature tensors [B, N, C] for each modality
            return_attention: If True, return attention weights
        
        Returns:
            fused_features: List of fused features for each modality
            attention_weights: List of attention weights (optional)
        """
        features = modality_features
        all_attentions = [] if return_attention else None
        
        for layer_idx in range(self.num_layers):
            # Cross-attention between modalities
            new_features = []
            layer_attentions = []
            
            for i, feat_i in enumerate(features):
                # Attend to all other modalities
                attended = []
                attns = []
                
                for j, feat_j in enumerate(features):
                    if i != j:
                        if return_attention:
                            attn_out, attn_weights = self.cross_attn_layers[layer_idx][i](
                                feat_i, feat_j, return_attention=True
                            )
                            attended.append(attn_out)
                            attns.append(attn_weights)
                        else:
                            attn_out = self.cross_attn_layers[layer_idx][i](feat_i, feat_j)
                            attended.append(attn_out)
                
                # Average attended features
                attended = torch.stack(attended, dim=0).mean(dim=0)
                
                # Residual connection and norm
                feat_i = feat_i + attended
                feat_i = self.norms[layer_idx * 2](feat_i)
                
                # Feed-forward
                feat_i = feat_i + self.ffns[layer_idx](feat_i)
                feat_i = self.norms[layer_idx * 2 + 1](feat_i)
                
                new_features.append(feat_i)
                if return_attention:
                    layer_attentions.append(attns)
            
            features = new_features
            if return_attention:
                all_attentions.append(layer_attentions)
        
        if return_attention:
            return features, all_attentions
        return features


class AttentionPooling(nn.Module):
    """
    Attention-based pooling
    
    Learns to pool features based on importance.
    """
    
    def __init__(self, dim):
        super(AttentionPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, N, C]
        
        Returns:
            pooled: Pooled features [B, C]
            attn_weights: Attention weights [B, N, 1]
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # [B, N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # [B, C]
        
        return pooled, attn_weights


if __name__ == "__main__":
    print("Testing attention modules...")
    
    # Test cross-modal attention
    print("\n1. Testing Cross-Modal Attention...")
    cross_attn = CrossModalAttention(dim=256, num_heads=8)
    x1 = torch.randn(2, 100, 256)  # Colposcopy features
    x2 = torch.randn(2, 100, 256)  # Histopathology features
    out, attn = cross_attn(x1, x2, return_attention=True)
    print(f"   Input x1 shape: {x1.shape}")
    print(f"   Input x2 shape: {x2.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention shape: {attn.shape}")
    
    # Test self-attention refinement
    print("\n2. Testing Self-Attention Refinement...")
    self_attn = SelfAttentionRefinement(dim=256, num_heads=8)
    x = torch.randn(2, 100, 256)
    out, attn = self_attn(x, return_attention=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention shape: {attn.shape}")
    
    # Test CBAM
    print("\n3. Testing CBAM...")
    cbam = CBAM(in_channels=256)
    x = torch.randn(2, 256, 14, 14)
    out, ch_attn, sp_attn = cbam(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Channel attention shape: {ch_attn.shape}")
    print(f"   Spatial attention shape: {sp_attn.shape}")
    
    # Test multi-head cross-attention fusion
    print("\n4. Testing Multi-Head Cross-Attention Fusion...")
    fusion = MultiHeadCrossAttentionFusion(dim=256, num_modalities=2, num_heads=8, num_layers=2)
    modality_features = [torch.randn(2, 100, 256), torch.randn(2, 100, 256)]
    fused, attns = fusion(modality_features, return_attention=True)
    print(f"   Input modalities: {len(modality_features)}")
    print(f"   Output modalities: {len(fused)}")
    print(f"   Fused feature shape: {fused[0].shape}")
    print(f"   Attention layers: {len(attns)}")
    
    # Test attention pooling
    print("\n5. Testing Attention Pooling...")
    attn_pool = AttentionPooling(dim=256)
    x = torch.randn(2, 100, 256)
    pooled, weights = attn_pool(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Pooled shape: {pooled.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    
    print("\n✓ All attention module tests passed!")
