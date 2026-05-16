from typing import Callable, List

import timm
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

_RESNET_STAGES = ("layer1", "layer2", "layer3", "layer4")
_HIGH_STAGES = ("layer3", "layer4")

_DEFAULT_TIMM_NAME = {
    "vit_tiny": "vit_tiny_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


def _wrap_resnet_blocks(
    model: nn.Module, layer_names: tuple, factory: Callable[[int], nn.Module]
) -> nn.Module:
    """在每个 BasicBlock 后串联 factory(channels) 得到的注意力模块。"""
    for layer_name in layer_names:
        layer = getattr(model, layer_name)
        blocks = []
        for block in layer:
            ch = block.bn2.num_features
            blocks.append(nn.Sequential(block, factory(ch)))
        setattr(model, layer_name, nn.Sequential(*blocks))
    return model


def _classifier_parameters(model: nn.Module) -> List[nn.Parameter]:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return list(model.fc.parameters())
    if hasattr(model, "get_classifier"):
        clf = model.get_classifier()
        if clf is not None:
            return list(clf.parameters())
    if hasattr(model, "head"):
        return list(model.head.parameters())
    raise ValueError(
        "Cannot resolve classifier params (expected ResNet .fc or timm get_classifier/head)."
    )


def build_model(cfg: dict) -> nn.Module:
    model_name = cfg["model_name"]
    pretrained = cfg["pretrained"]
    attention = cfg.get("attention", "none")

    if model_name in _DEFAULT_TIMM_NAME:
        if attention != "none":
            raise ValueError(f"attention must be 'none' for model_name={model_name}, got {attention}")
        timm_name = cfg.get("timm_model_name") or _DEFAULT_TIMM_NAME[model_name]
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=cfg["num_classes"],
        )
        return model

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if attention == "se":
        model = _wrap_resnet_blocks(model, _RESNET_STAGES, lambda c: SEBlock(c))
    elif attention == "se_high":
        model = _wrap_resnet_blocks(model, _HIGH_STAGES, lambda c: SEBlock(c))
    elif attention == "cbam":
        model = _wrap_resnet_blocks(model, _RESNET_STAGES, lambda c: CBAM(c))
    elif attention == "cbam_high":
        model = _wrap_resnet_blocks(model, _HIGH_STAGES, lambda c: CBAM(c))
    elif attention != "none":
        raise ValueError(f"Unsupported attention type: {attention}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, cfg["num_classes"])
    return model


def build_optimizer(cfg: dict, model: nn.Module):
    head_params = _classifier_parameters(model)
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    param_groups = [
        {"params": backbone_params, "lr": cfg["backbone_lr"]},
        {"params": head_params, "lr": cfg["head_lr"]},
    ]

    optimizer_name = cfg.get("optimizer", "sgd").lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 1e-4))
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(cfg: dict, optimizer):
    scheduler_name = cfg.get("scheduler", "cosine").lower()
    epochs = int(cfg["epochs"])

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if scheduler_name == "cosine_warmup":
        warmup = int(cfg.get("warmup_epochs", 5))
        if warmup < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if warmup >= epochs:
            raise ValueError("warmup_epochs must be < epochs for cosine_warmup")
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warm = LinearLR(
            optimizer,
            start_factor=float(cfg.get("warmup_start_factor", 1e-3)),
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_steps = max(1, epochs - warmup)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])

    if scheduler_name == "step":
        step_size = cfg.get("step_size", 10)
        gamma = cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
