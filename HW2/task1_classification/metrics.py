import torch


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float().sum().item()
    total = targets.numel()
    return correct / max(total, 1)

