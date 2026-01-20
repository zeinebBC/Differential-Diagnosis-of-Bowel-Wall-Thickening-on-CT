import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardLoss(nn.Module):
    """
    Multi-class Jaccard Loss (1 - soft IoU) for B x C x H x W x D data,
 
    """
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
       
        prob = F.softmax(input, dim=1)
        print("prob", prob)
        num_classes = prob.size(1)
        losses = []

        prob_flat = prob.view(prob.size(0), num_classes, -1)
        target_flat = target.view(target.size(0), -1)

        
        for c in range(num_classes):
            
            target_c = (target_flat == c).float()
            input_c = prob_flat[:, c, :]

            intersection = (input_c * target_c).sum(dim=1) 
            total = (input_c + target_c).sum(dim=1) 
    
            union = total - intersection
            
            IoU = (intersection + self.smooth) / (union + self.smooth)

            losses.append(1.0 - IoU)
        
    
        losses_stacked = torch.stack(losses)
        loss = losses_stacked.mean()
        
        return loss



class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2., eps=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
       

    def forward(self, y_pred, y_true):
        prob = F.softmax(y_pred, dim=1)
        num_classes = prob.size(1)
        losses = []
        prob_flat = prob.view(prob.size(0), num_classes, -1)
        target_flat = y_true.view(y_true.size(0), -1)
        for c in range(num_classes):
            input_c = prob_flat[:, c, :]
            target_c = (target_flat == c).float()
            dim_to_sum = 1 

            t_p = (input_c * target_c).sum(dim=dim_to_sum)  
     
            f_p = ((1.0 - target_c) * input_c).sum(dim=dim_to_sum) 

            f_n = (target_c * (1.0 - input_c)).sum(dim=dim_to_sum) 
            
          
            tversky = (t_p + self.eps) / (t_p + self.alpha * f_p + self.beta * f_n + self.eps)

            focal_tversky = (1.0 - tversky)**self.gamma
            losses.append(focal_tversky)
        
       
        losses_stacked = torch.stack(losses)
        loss = losses_stacked.mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss based on Cross-Entropy
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # CE loss per pixel (no reduction)
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')

        # Probability of the true class
        p = torch.softmax(y_pred, dim=1)
        pt = torch.gather(p, 1, y_true.unsqueeze(1)).squeeze(1)

        # Focal loss weighting
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
