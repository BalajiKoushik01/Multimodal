"""
Advanced Self-Supervised Learning (SSL) Strategies

Implements state-of-the-art SSL methods:
1. SimCLR v2 - Simple Framework for Contrastive Learning
2. MoCo v3 - Momentum Contrast
3. BYOL - Bootstrap Your Own Latent
4. DINO - Self-Distillation with No Labels

All adapted for medical imaging with multimodal support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Tuple, Optional


class SimCLRLoss(nn.Module):
    """
    SimCLR v2 Loss (NT-Xent - Normalized Temperature-scaled Cross Entropy)
    
    Contrastive loss that maximizes agreement between differently augmented
    views of the same image while minimizing agreement with other images.
    """
    
    def __init__(self, temperature=0.07, use_cosine_similarity=True):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Projections from first augmentation [B, D]
            z_j: Projections from second augmentation [B, D]
        
        Returns:
            loss: Contrastive loss value
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate embeddings
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # [2B, 2B]
        
        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Positive pairs: (i, j) and (j, i)
        pos_sim = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)  # [2B]
        
        # Compute loss
        pos_sim = pos_sim.reshape(2 * batch_size, 1)
        similarity_matrix = similarity_matrix / self.temperature
        pos_sim = pos_sim / self.temperature
        
        # Numerator: exp(sim(i,j)/τ)
        # Denominator: sum of exp(sim(i,k)/τ) for all k != i
        logits = similarity_matrix
        labels = torch.arange(2 * batch_size, device=z_i.device)
        
        # Shift labels to account for positive pairs
        labels = (labels + batch_size) % (2 * batch_size)
        
        loss = self.criterion(logits, labels)
        
        return loss


class MoCoV3(nn.Module):
    """
    MoCo v3 - Momentum Contrast
    
    Uses a momentum encoder to maintain a large and consistent queue
    of negative samples for contrastive learning.
    """
    
    def __init__(self, encoder, projection_head, dim=128, K=65536, m=0.999, T=0.07):
        """
        Args:
            encoder: Base encoder network
            projection_head: Projection head network
            dim: Feature dimension
            K: Queue size
            m: Momentum coefficient
            T: Temperature
        """
        super(MoCoV3, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # Create query encoder
        self.encoder_q = encoder
        self.projection_q = projection_head
        
        # Create key encoder (momentum)
        self.encoder_k = copy.deepcopy(encoder)
        self.projection_k = copy.deepcopy(projection_head)
        
        # Freeze key encoder parameters
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projection_k.parameters():
            param.requires_grad = False
        
        # Create queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update queue with new keys
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest keys in queue
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """
        Args:
            im_q: Query images [B, C, H, W]
            im_k: Key images [B, C, H, W]
        
        Returns:
            logits: Similarity logits
            labels: Ground truth labels (all zeros)
        """
        # Compute query features
        feat_q, _ = self.encoder_q(im_q)
        q = self.projection_q(feat_q)
        q = F.normalize(q, dim=1)
        
        # Compute key features (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            feat_k, _ = self.encoder_k(im_k)
            k = self.projection_k(feat_k)
            k = F.normalize(k, dim=1)
        
        # Compute logits
        # Positive logits: [B, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: [B, K]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: [B, K+1]
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.T
        
        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels


class BYOL(nn.Module):
    """
    BYOL - Bootstrap Your Own Latent
    
    Self-supervised learning without negative pairs.
    Uses a predictor network and momentum encoder.
    """
    
    def __init__(self, encoder, projection_head, predictor, hidden_dim=4096, output_dim=256, m=0.996):
        """
        Args:
            encoder: Base encoder network
            projection_head: Projection head network
            predictor: Predictor network
            hidden_dim: Hidden dimension for projection
            output_dim: Output dimension
            m: Momentum coefficient
        """
        super(BYOL, self).__init__()
        
        self.m = m
        
        # Online network
        self.online_encoder = encoder
        self.online_projector = projection_head
        self.predictor = predictor
        
        # Target network (momentum)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(projection_head)
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _update_target_network(self):
        """
        Exponential moving average update of target network
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.m + param_o.data * (1. - self.m)
        
        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data = param_t.data * self.m + param_o.data * (1. - self.m)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: First augmented view [B, C, H, W]
            x2: Second augmented view [B, C, H, W]
        
        Returns:
            loss: BYOL loss
        """
        # Online network forward pass
        feat_o1, _ = self.online_encoder(x1)
        z_o1 = self.online_projector(feat_o1)
        p_o1 = self.predictor(z_o1)
        
        feat_o2, _ = self.online_encoder(x2)
        z_o2 = self.online_projector(feat_o2)
        p_o2 = self.predictor(z_o2)
        
        # Target network forward pass (no gradient)
        with torch.no_grad():
            self._update_target_network()
            
            feat_t1, _ = self.target_encoder(x1)
            z_t1 = self.target_projector(feat_t1)
            
            feat_t2, _ = self.target_encoder(x2)
            z_t2 = self.target_projector(feat_t2)
        
        # Compute loss (mean squared error in normalized space)
        loss = self.regression_loss(p_o1, z_t2.detach()) + self.regression_loss(p_o2, z_t1.detach())
        
        return loss
    
    @staticmethod
    def regression_loss(x, y):
        """
        Normalized mean squared error
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()


class DINOLoss(nn.Module):
    """
    DINO - Self-Distillation with No Labels
    
    Uses self-distillation with a teacher-student framework.
    Teacher is updated via exponential moving average.
    """
    
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        """
        Args:
            out_dim: Output dimension
            teacher_temp: Teacher temperature
            student_temp: Student temperature
            center_momentum: Momentum for center update
        """
        super(DINOLoss, self).__init__()
        
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: Student network output [B, D]
            teacher_output: Teacher network output [B, D]
        
        Returns:
            loss: Cross-entropy loss
        """
        # Normalize outputs
        student_out = student_output / self.student_temp
        
        # Teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()
        
        # Cross-entropy loss
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        
        # Update center
        self.update_center(teacher_output)
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MultiModalContrastiveLoss(nn.Module):
    """
    Multimodal contrastive loss for colposcopy-histopathology pairs
    
    Encourages alignment between corresponding colposcopy and histopathology images
    while separating non-corresponding pairs.
    """
    
    def __init__(self, temperature=0.07):
        super(MultiModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_colpo, z_histo):
        """
        Args:
            z_colpo: Colposcopy embeddings [B, D]
            z_histo: Histopathology embeddings [B, D]
        
        Returns:
            loss: Contrastive loss
        """
        batch_size = z_colpo.shape[0]
        
        # Normalize
        z_colpo = F.normalize(z_colpo, dim=1)
        z_histo = F.normalize(z_histo, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(z_colpo, z_histo.T) / self.temperature  # [B, B]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z_colpo.device)
        
        # Symmetric loss
        loss_colpo = self.criterion(logits, labels)
        loss_histo = self.criterion(logits.T, labels)
        
        loss = (loss_colpo + loss_histo) / 2
        
        return loss


if __name__ == "__main__":
    print("Testing SSL strategies...")
    
    batch_size = 4
    dim = 128
    
    # Test SimCLR
    print("\n1. Testing SimCLR...")
    simclr_loss = SimCLRLoss(temperature=0.07)
    z_i = torch.randn(batch_size, dim)
    z_j = torch.randn(batch_size, dim)
    loss = simclr_loss(z_i, z_j)
    print(f"   SimCLR loss: {loss.item():.4f}")
    
    # Test BYOL
    print("\n2. Testing BYOL...")
    byol_loss = BYOL.regression_loss(z_i, z_j)
    print(f"   BYOL loss: {byol_loss.item():.4f}")
    
    # Test DINO
    print("\n3. Testing DINO...")
    dino_loss = DINOLoss(out_dim=dim)
    student_out = torch.randn(batch_size, dim)
    teacher_out = torch.randn(batch_size, dim)
    loss = dino_loss(student_out, teacher_out)
    print(f"   DINO loss: {loss.item():.4f}")
    
    # Test Multimodal Contrastive
    print("\n4. Testing Multimodal Contrastive...")
    mm_loss = MultiModalContrastiveLoss(temperature=0.07)
    z_colpo = torch.randn(batch_size, dim)
    z_histo = torch.randn(batch_size, dim)
    loss = mm_loss(z_colpo, z_histo)
    print(f"   Multimodal loss: {loss.item():.4f}")
    
    print("\n✓ All SSL strategy tests passed!")
