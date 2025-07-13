import torch.nn.functional as F
import torch.nn as nn
import torch

class DisentangleLoss(nn.Module):
    def __init__(self, loss_weight=0.2, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, homo_feat, hetero_feat):
        h_global = homo_feat.mean(dim=1)    # (B, N, D)
        e_global = hetero_feat.mean(dim=1)  # (B, N, D)
        
        h_norm = F.normalize(h_global, p=2, dim=-1)  # (B, N, D)
        e_norm = F.normalize(e_global, p=2, dim=-1)
        
        similarity = torch.bmm(h_norm, e_norm.transpose(1,2))  # (B, N, N)
        
        on_diag = similarity.diagonal(dim1=1, dim2=2).pow(2).mean()
        off_diag = similarity.flatten(1)[:, :-1].view(similarity.size(0),-1).pow(2).mean()
        
        loss = (off_diag - 0.5*on_diag) 
        
        return self.loss_weight * loss