import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """
    Center Loss for Anomaly Detection.
    Maintains a 'center' in the embedding space and pulls all normal representations towards it.
    
    Formula: 0.5 * || h - c ||^2
    """
    def __init__(self, num_classes=1, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        # In One-Class Anomaly Detection, we often have just 1 center for "Normal"
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size). 
                    For anomaly detection training, we assume everything is normal (label 0).
        """
        batch_size = x.size(0)
        
        # Calculate squared distance from center
        # One-Class case: just distance to centers[0]
        # x: (B, D)
        # centers: (1, D)
        
        # distmat = (x - c)^2
        # expanded_centers = self.centers.expand(batch_size, -1)
        # dist = torch.pow(x - expanded_centers, 2).sum(dim=1)
        
        # More robust implementation for general case (if we wanted multiple clusters)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        # For one-class (normal data training), we want to minimize distance to the ONLY center.
        # We assume labels are all 0 (Normal) during training phase.
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        
        # We only care about the distance to the '0' center
        # masks = torch.eq(labels.unsqueeze(1).expand(batch_size, self.num_classes), classes.unsqueeze(0))
        
        # Simplified for 1-class:
        # dist = distmat[:, 0]
        
        # But let's stick to standard implementation to be safe if we expand later
        if labels is None:
             # Assume all 0
             labels = torch.zeros(batch_size, dtype=torch.long, device=x.device)
             
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
