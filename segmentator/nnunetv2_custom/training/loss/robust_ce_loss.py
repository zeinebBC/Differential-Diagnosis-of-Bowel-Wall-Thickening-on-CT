import torch
from torch import nn, Tensor
import numpy as np


import torch.nn.functional as F



class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized Cross Entropy Loss (GCE)
    Drop-in replacement for nn.CrossEntropyLoss.

    Args:
        q (float): Noise-robustness parameter in (0, 1]. 
                   Smaller q -> more robustness to label noise.
                   q = 1 reduces to standard CrossEntropyLoss.
        reduction (str): 'mean' or 'sum'
    """
    def __init__(self, q: float = 0.7, reduction: str = 'mean'):
        super().__init__()
        assert 0 < q <= 1, "q must be in (0, 1]"
        self.q = q
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: logits tensor of shape [B, C, *]
        target: ground truth tensor of shape [B, *] or [B, 1, *]
        """
        if target.ndim == input.ndim:
            assert target.shape[1] == 1, f"Unexpected target shape {target.shape}"
            target = target[:, 0]

        # Compute softmax probabilities
        probs = F.softmax(input, dim=1)

        # Gather probability of the correct class for each pixel
        target_long = target.long()
        p = probs.gather(1, target_long.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)

        # Apply GCE formula: L = (1 - p^q) / q
        loss = (1 - p.pow(self.q)) / self.q

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
