import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _resolve_images_dir(root: str) -> str:
    images_root = os.path.join(root, "images")
    nested = os.path.join(images_root, "images")
    return nested if os.path.isdir(nested) else images_root


def _resolve_ann_dir(root: str) -> str:
    ann_root = os.path.join(root, "annotations")
    nested = os.path.join(ann_root, "annotations")
    if os.path.exists(os.path.join(ann_root, "trainval.txt")):
        return ann_root
    if os.path.exists(os.path.join(nested, "trainval.txt")):
        return nested
    return ann_root


def _resolve_trimaps_dir(ann_dir: str) -> str:
    trimaps_dir = os.path.join(ann_dir, "trimaps")
    if not os.path.isdir(trimaps_dir):
        raise FileNotFoundError(
            f"Cannot find trimaps directory: {trimaps_dir}. "
            "Please ensure Oxford-IIIT segmentation trimaps are extracted."
        )
    return trimaps_dir


class OxfordPetSegmentation(Dataset):
    def __init__(self, root: str, split: str, img_size: int = 256, augment: bool = False):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.images_dir = _resolve_images_dir(root)
        self.ann_dir = _resolve_ann_dir(root)
        self.trimaps_dir = _resolve_trimaps_dir(self.ann_dir)
        self.samples = self._load_samples()

    def _split_file(self) -> str:
        if self.split == "trainval":
            return os.path.join(self.ann_dir, "trainval.txt")
        if self.split == "test":
            return os.path.join(self.ann_dir, "test.txt")
        raise ValueError(f"Unsupported split: {self.split}")

    def _load_samples(self):
        split_file = self._split_file()
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Cannot find split file: {split_file}")

        samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name = line.split()[0]
                image_path = self._resolve_image_path(name)
                mask_path = os.path.join(self.trimaps_dir, f"{name}.png")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Cannot find trimap file: {mask_path}")
                samples.append((image_path, mask_path))
        return samples

    def _resolve_image_path(self, name: str) -> str:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(self.images_dir, f"{name}{ext}")
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Cannot find image file for sample: {name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        if self.augment and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)

        # Oxford-IIIT trimap labels are {1,2,3}. We shift to {0,1,2}.
        mask_np = np.array(mask, dtype=np.int64) - 1
        mask_np = np.clip(mask_np, 0, 2)
        mask_tensor = torch.from_numpy(mask_np).long()
        return image, mask_tensor


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    trainval_train_ds = OxfordPetSegmentation(
        cfg["data_root"], split="trainval", img_size=cfg["img_size"], augment=True
    )
    trainval_val_ds = OxfordPetSegmentation(
        cfg["data_root"], split="trainval", img_size=cfg["img_size"], augment=False
    )

    total = len(trainval_train_ds)
    train_size = int(total * 0.8)
    g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    indices = torch.randperm(total, generator=g).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = Subset(trainval_train_ds, train_indices)
    val_ds = Subset(trainval_val_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader
