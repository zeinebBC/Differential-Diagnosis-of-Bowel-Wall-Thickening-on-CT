import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss, GeneralizedCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from nnunetv2.training.loss.utils_losses import FocalLoss, FocalTverskyLoss, JaccardLoss


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss, weight_penalty=1):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_penalty = weight_penalty

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = GeneralizedCrossEntropyLoss(**ce_kwargs )
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, distance_map:torch.Tensor=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        """"
        penalty_loss = 0
        if distance_map is not None and self.weight_penalty > 0:
            prob_fg = torch.softmax(net_output, dim=1)[:, 1:2]  # assume channel 1 = FG
            dist_map_norm = torch.log1p(distance_map).to(prob_fg.device)
            penalty_loss = (prob_fg * dist_map_norm).mean()
        """
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss # + self.weight_penalty * penalty_loss
        return result




class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class CE_and_FocalTverskyLoss(nn.Module):
    def __init__(self, focal_tversky_kwargs, ce_kwargs, weight_focal_tversky=1, weight_ce=1):
        super().__init__()
        self.weight_focal_tversky = weight_focal_tversky
        self.weight_ce = weight_ce

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ftv = FocalTverskyLoss( **focal_tversky_kwargs)

    def forward(self, net_output, target):
        ftv_loss = self.ftv(net_output, target) if self.weight_focal_tversky else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce else 0
        return self.weight_ce * ce_loss + self.weight_focal_tversky * ftv_loss
    
class CE_and_FocalTversky_withPenalty(nn.Module):
    def __init__(self, focal_tversky_kwargs, ce_kwargs,
                 weight_focal_tversky=1, weight_ce=1, weight_penalty=1):
        """
        Combines Cross-Entropy, Focal Tversky Loss, and an optional distance map penalty.
        :param focal_tversky_kwargs: kwargs for FocalTverskyLoss
        :param ce_kwargs: kwargs for RobustCrossEntropyLoss
        :param weight_focal_tversky: weight for Focal Tversky Loss
        :param weight_ce: weight for Cross-Entropy
        :param weight_penalty: weight for distance map penalty
        """
        super().__init__()
        self.weight_focal_tversky = weight_focal_tversky
        self.weight_ce = weight_ce
        self.weight_penalty = weight_penalty

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ftv = FocalTverskyLoss(**focal_tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, distance_map: torch.Tensor = None):
        # Compute Focal Tversky loss
        ftv_loss = self.ftv(net_output, target) if self.weight_focal_tversky != 0 else 0
        # Compute Cross-Entropy loss
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0

        # Compute optional distance map penalty
        penalty_loss = 0
        if distance_map is not None and self.weight_penalty > 0:
            # assume channel 1 = foreground
            prob_fg = torch.softmax(net_output, dim=1)[:, 1:2]
            dist_map_norm = torch.log1p(distance_map).to(prob_fg.device)
            penalty_loss = (prob_fg * dist_map_norm).mean()

        # Combine all losses
        total_loss = (self.weight_ce * ce_loss +
                      self.weight_focal_tversky * ftv_loss +
                      self.weight_penalty * penalty_loss)
        return total_loss


class Jaccard_and_CE_loss(nn.Module):
    def __init__(self, jaccard_kwargs, ce_kwargs, weight_ce=1, weight_jaccard=1):
        super().__init__()
        self.weight_jaccard = weight_jaccard
        self.weight_ce = weight_ce

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.jaccard = JaccardLoss( **jaccard_kwargs)

    def forward(self, net_output, target):
        jaccard_loss = self.jaccard(net_output, target) if self.weight_jaccard else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce else 0
        return self.weight_ce * ce_loss + self.weight_jaccard * jaccard_loss


class DC_and_FocalLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1):
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
   

        self.focal = FocalLoss(**focal_kwargs)
        self.dc = SoftDiceLoss( **soft_dice_kwargs)

    def forward(self, net_output, target):
     
        dc_loss = self.dc(net_output, target) if self.weight_dice else 0
        focal_loss = self.focal(net_output, target) if self.weight_focal else 0

        return self.weight_focal * focal_loss + self.weight_dice * dc_loss
