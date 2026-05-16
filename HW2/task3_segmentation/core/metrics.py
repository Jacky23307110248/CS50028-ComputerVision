import torch


@torch.no_grad()
def compute_miou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3) -> float:
    preds = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = targets == cls
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))
