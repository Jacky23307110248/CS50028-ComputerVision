import torch

from core.metrics import compute_miou


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    total_steps = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_miou += compute_miou(logits.detach(), masks.detach())
        total_steps += 1

    return {
        "loss": total_loss / max(total_steps, 1),
        "miou": total_miou / max(total_steps, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_steps = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_miou += compute_miou(logits, masks)
        total_steps += 1

    return {
        "loss": total_loss / max(total_steps, 1),
        "miou": total_miou / max(total_steps, 1),
    }
