import os
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OxfordPetClassification(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images_dir = self._resolve_images_dir(root)
        self.ann_dir = self._resolve_ann_dir(root)
        self.samples, self.class_to_idx = self._load_samples()

    @staticmethod
    def _resolve_images_dir(root: str) -> str:
        """
        Support both common layouts:
        1) root/images/*.jpg
        2) root/images/images/*.jpg
        """
        images_root = os.path.join(root, "images")
        nested_images = os.path.join(images_root, "images")

        if os.path.isdir(nested_images):
            return nested_images
        return images_root

    @staticmethod
    def _resolve_ann_dir(root: str) -> str:
        """
        Support both common layouts:
        1) root/annotations/trainval.txt
        2) root/annotations/annotations/trainval.txt
        """
        ann_root = os.path.join(root, "annotations")
        nested_ann = os.path.join(ann_root, "annotations")

        if os.path.exists(os.path.join(ann_root, "trainval.txt")):
            return ann_root
        if os.path.exists(os.path.join(nested_ann, "trainval.txt")):
            return nested_ann
        return ann_root

    def _split_file(self) -> str:
        if self.split == "trainval":
            return os.path.join(self.ann_dir, "trainval.txt")
        if self.split == "test":
            return os.path.join(self.ann_dir, "test.txt")
        raise ValueError(f"Unsupported split: {self.split}")

    def _load_samples(self) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        split_file = self._split_file()
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Cannot find split file: {split_file}. "
                "Expected either 'root/annotations/*.txt' or "
                "'root/annotations/annotations/*.txt'."
            )

        lines = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        class_names = set()
        parsed = []
        for line in lines:
            # Format: image_name class_id species breed_id
            parts = line.split()
            image_name = parts[0]
            class_name = image_name.rsplit("_", 1)[0]
            class_names.add(class_name)
            parsed.append((image_name, class_name))

        sorted_classes = sorted(class_names)
        class_to_idx = {name: i for i, name in enumerate(sorted_classes)}

        samples: List[Tuple[str, int]] = []
        for image_name, class_name in parsed:
            image_path = self._resolve_image_path(image_name)
            label = class_to_idx[class_name]
            samples.append((image_path, label))
        return samples, class_to_idx

    def _resolve_image_path(self, image_name: str) -> str:
        candidates = [
            os.path.join(self.images_dir, f"{image_name}.jpg"),
            os.path.join(self.images_dir, f"{image_name}.jpeg"),
            os.path.join(self.images_dir, f"{image_name}.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        # Keep previous behavior (raise at __getitem__) but with clearer path candidate.
        return candidates[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def _build_train_val_subsets(cfg: dict, train_tf, eval_tf):
    dataset_root = cfg["data_root"]
    full_train_ds = OxfordPetClassification(dataset_root, split="trainval", transform=train_tf)
    full_val_ds = OxfordPetClassification(dataset_root, split="trainval", transform=eval_tf)

    total = len(full_train_ds)
    if total < 2:
        raise ValueError("trainval split has too few samples for train/val split")

    val_ratio = float(cfg.get("val_ratio", 0.1))
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError(f"val_ratio={val_ratio} leaves no training sample")

    split_seed = int(cfg.get("split_seed", cfg.get("seed", 42)))
    g = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total, generator=g).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_val_ds, val_indices)
    return train_ds, val_ds


def build_dataloaders(cfg: dict):
    train_tf, eval_tf = build_transforms(cfg["img_size"])
    train_ds, val_ds = _build_train_val_subsets(cfg, train_tf, eval_tf)

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


def build_test_loader(cfg: dict):
    _, eval_tf = build_transforms(cfg["img_size"])
    dataset_root = cfg["data_root"]
    test_ds = OxfordPetClassification(dataset_root, split="test", transform=eval_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    return test_loader

