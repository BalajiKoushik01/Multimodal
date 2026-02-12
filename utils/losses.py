
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features_1, features_2):
        batch_size = features_1.shape[0]
        concat_features = torch.cat([features_1, features_2], dim=0)
        
        # Similarity matrix
        sim_matrix = torch.matmul(concat_features, concat_features.T) / self.temperature
        
        # Mask out self-contrast cases
        mask = torch.eye(2 * batch_size).to(features_1.device)
        sim_matrix = sim_matrix.masked_fill(mask == 1, float('-inf'))
        
        # Positive pairs (diagonal offset by batch_size)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(features_1.device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class PrototypeLoss(nn.Module):
    def __init__(self, num_prototypes=10, feature_dim=128):
        super(PrototypeLoss, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        
    def forward(self, features, labels=None):
        # Assign features to closest prototype via cosine similarity
        norm_prototypes = F.normalize(self.prototypes, dim=1)
        norm_features = F.normalize(features, dim=1)
        
        scores = torch.mm(norm_features, norm_prototypes.t())
        
        # If labels provided (weak supervision), encourage features of same class to map to same proto
        if labels is not None:
             pass # Implement supervised prototype alignment here
             
        # Example: maximize agreement between paired views over prototypes (SwAV style)
        q = F.softmax(scores / 0.1, dim=1) # soft assignment
        
        loss = -torch.mean(torch.sum(q * F.log_softmax(scores / 0.1, dim=1), dim=1)) # simple entropy min
        return loss

class TotalLoss(nn.Module):
    def __init__(self, weights={'contrastive': 1.0, 'prototype': 0.5, 'cls': 1.0}):
        super(TotalLoss, self).__init__()
        self.contrastive = InfoNCELoss()
        self.prototype = PrototypeLoss()
        self.cls = nn.CrossEntropyLoss()
        self.weights = weights

    def forward(self, out_c, out_h, labels_malignancy=None, labels_severity=None):
        loss_con = self.contrastive(out_c, out_h)
        loss_proto = self.prototype(out_c) + self.prototype(out_h)
        
        total_loss = self.weights['contrastive'] * loss_con + self.weights['prototype'] * loss_proto
        
        if labels_malignancy is not None:
             # Assume output includes classification heads 
             # For now skipping this part in loss wrapper as it typically comes from trainer
             pass
             
        return total_loss, {'contrastive': loss_con, 'prototype': loss_proto}
