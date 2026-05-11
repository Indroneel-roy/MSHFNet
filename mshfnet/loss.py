# -*- coding: utf-8 -*-
"""Loss functions for MSHFNet.

Combined loss formula
----------------------
  L = 0.4 × CE(class-weighted) + 0.4 × Dice + 0.2 × Boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Class weights for Synapse (9 classes: background + 8 organs)
# Inverse-frequency design: small / hard organs get higher weight.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_WEIGHTS = torch.tensor([
    0.5,   # 0: background   — very frequent, penalise less
    1.5,   # 1: aorta        — tubular, moderate
    3.0,   # 2: gallbladder  — tiny, often missing → boost
    1.2,   # 3: spleen       — large, easier
    1.5,   # 4: left kidney  — moderate
    0.8,   # 5: liver        — large, easiest organ
    2.0,   # 6: right kidney — consistently weak → boost
    1.2,   # 7: stomach      — variable shape
    2.0,   # 8: pancreas     — small, hard
])


# ─────────────────────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft Dice Loss averaged over all classes.

    Parameters
    ----------
    num_classes : int   — number of segmentation classes.
    smooth      : float — Laplace smoothing to avoid division by zero.
    """

    def __init__(self, num_classes: int = 9, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions     = torch.softmax(predictions, dim=1)
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        dice_per_class = []
        for c in range(self.num_classes):
            pred_c       = predictions[:, c]
            target_c     = targets_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            union        = pred_c.sum() + target_c.sum()
            dice         = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)

        return 1 - torch.stack(dice_per_class).mean()


class BoundaryLoss(nn.Module):
    """Boundary-weighted MSE loss targeting HD95 improvement.

    Parameters
    ----------
    num_classes : int — number of segmentation classes.
                        Background (class 0) is skipped.
    """

    def __init__(self, num_classes: int = 9):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        # 3×3 structuring element for morphological ops
        self.kernel = torch.ones(1, 1, 3, 3)

    def extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Return binary boundary map via dilation − erosion.

        Parameters
        ----------
        mask : (B, H, W) float tensor — binary organ mask.

        Returns
        -------
        boundary : (B, H, W) float tensor in [0, 1].
        """
        m      = mask.unsqueeze(1)                              # (B, 1, H, W)
        kernel = self.kernel.to(mask.device)
        dilate = F.conv2d(m,       kernel, padding=1).clamp(0, 1)
        erode  = F.conv2d(1 - m,  kernel, padding=1).clamp(0, 1)
        erode  = 1 - erode
        return (dilate - erode).clamp(0, 1).squeeze(1)         # (B, H, W)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_soft     = torch.softmax(predictions, dim=1)
        boundary_loss = 0.0
        for c in range(1, self.num_classes):                    # skip background
            pred_c   = pred_soft[:, c]
            target_c = (targets == c).float()
            boundary = self.extract_boundary(target_c)
            # MSE penalised only at boundary pixels
            boundary_loss += (boundary * (pred_c - target_c).pow(2)).mean()
        return boundary_loss / (self.num_classes - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss
# L = 0.4 × CE(weighted) + 0.4 × Dice + 0.2 × Boundary
# ─────────────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """Weighted combination of class-weighted CE, Dice, and Boundary losses.

    Parameters
    ----------
    num_classes   : int            — number of segmentation classes.
    class_weights : torch.Tensor   — 1-D weight tensor of length num_classes.
                                     Pass None to use uniform weights.
    """

    def __init__(self, num_classes: int = 9, class_weights: torch.Tensor = None):
        super(CombinedLoss, self).__init__()
        self.ce_loss       = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss     = DiceLoss(num_classes)
        self.boundary_loss = BoundaryLoss(num_classes)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce       = self.ce_loss(predictions, targets)
        dice     = self.dice_loss(predictions, targets)
        boundary = self.boundary_loss(predictions, targets)
        return 0.4 * ce + 0.4 * dice + 0.2 * boundary


# ─────────────────────────────────────────────────────────────────────────────
# Deep Supervision Loss
# Applies CombinedLoss at the main output and three auxiliary decoder heads.
# ─────────────────────────────────────────────────────────────────────────────
class DeepSupervisionLoss(nn.Module):
    """CombinedLoss applied at four decoder scales with configurable weights.

    Parameters
    ----------
    num_classes   : int           — number of segmentation classes.
    weights       : list[float]   — loss weights for [main, ds1, ds2, ds3].
                                    Default: [1.0, 0.4, 0.2, 0.1].
    class_weights : torch.Tensor  — per-class weights forwarded to CombinedLoss.
    """

    def __init__(
        self,
        num_classes:   int           = 9,
        weights:       list          = None,
        class_weights: torch.Tensor  = None,
    ):
        super(DeepSupervisionLoss, self).__init__()
        self.weights = weights if weights is not None else [1.0, 0.4, 0.2, 0.1]
        self.loss_fn = CombinedLoss(num_classes, class_weights=class_weights)

    def forward(self, outputs: tuple, targets: torch.Tensor):
        """
        Parameters
        ----------
        outputs : tuple of (main, ds1, ds2, ds3) tensors — decoder predictions.
        targets : (B, H, W) long tensor — ground-truth segmentation labels.

        Returns
        -------
        total     : scalar — weighted sum across all scales.
        loss_main : scalar — main output loss (used for monitoring).
        loss_ds1  : scalar — first auxiliary loss (used for monitoring).
        """
        main, ds1, ds2, ds3 = outputs

        loss_main = self.loss_fn(main, targets)
        loss_ds1  = self.loss_fn(ds1,  targets)
        loss_ds2  = self.loss_fn(ds2,  targets)
        loss_ds3  = self.loss_fn(ds3,  targets)

        total = (
            self.weights[0] * loss_main +
            self.weights[1] * loss_ds1  +
            self.weights[2] * loss_ds2  +
            self.weights[3] * loss_ds3
        )
        return total, loss_main, loss_ds1
