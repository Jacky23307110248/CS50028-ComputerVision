from .unet import UNet


def build_model(name: str = "unet", num_classes: int = 3):
    if name != "unet":
        raise ValueError(f"Unsupported model: {name}")
    return UNet(in_channels=3, num_classes=num_classes)
