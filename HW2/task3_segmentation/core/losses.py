import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dim=dims)
        union = torch.sum(probs + targets_oh, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, loss_type: str, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "ce":
            return self.ce(logits, targets)
        if self.loss_type == "dice":
            return self.dice(logits, targets)
        if self.loss_type == "ce_dice":
            return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")
