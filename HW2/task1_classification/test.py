import argparse
import os
import sys

import torch

from dataset import build_dataloaders, build_test_loader
from losses import build_classification_loss
from model_factory import build_model
from utils import load_yaml, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Task1 environment and pipeline smoke test.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_resnet18.yaml",
        help="Config path relative to task1_classification/",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use cuda for ROCm too (PyTorch keeps cuda API).",
    )
    parser.add_argument(
        "--pretrained-mode",
        type=str,
        default="off",
        choices=["off", "on", "config"],
        help="off avoids downloading weights in smoke test.",
    )
    parser.add_argument("--max-batches", type=int, default=1, help="How many batches to test for each loader.")
    return parser.parse_args()


def resolve_cfg(args) -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)
    cfg = load_yaml(config_path)

    default_candidates = [
        os.path.join(base_dir, "..", "Oxford-IIIT"),
        os.path.join(base_dir, "dataset"),
    ]
    default_data_root = None
    for candidate in default_candidates:
        candidate_abs = os.path.abspath(candidate)
        if os.path.isdir(candidate_abs):
            default_data_root = candidate_abs
            break
    if default_data_root is None:
        default_data_root = os.path.abspath(default_candidates[0])

    cfg.setdefault("data_root", default_data_root)
    if not os.path.isabs(cfg["data_root"]):
        cfg["data_root"] = os.path.join(base_dir, cfg["data_root"])

    cfg.setdefault("img_size", 224)
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("seed", args.seed)
    cfg.setdefault("val_ratio", 0.1)
    cfg.setdefault("split_seed", cfg.get("seed", 42))
    cfg.setdefault("num_classes", 37)
    cfg.setdefault("model_name", "resnet18")
    cfg.setdefault("attention", "none")

    if args.pretrained_mode == "off":
        cfg["pretrained"] = False
    elif args.pretrained_mode == "on":
        cfg["pretrained"] = True
    else:
        cfg.setdefault("pretrained", True)
    return cfg


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_loader_check(name: str, loader, model, criterion, device: torch.device, max_batches: int):
    model.eval()
    checked = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            checked += 1
            print(
                f"[{name}] batch={checked} shape={tuple(images.shape)} "
                f"logits={tuple(logits.shape)} loss={loss.item():.4f} acc={acc:.4f}"
            )
            if checked >= max_batches:
                break
    if checked == 0:
        raise RuntimeError(f"{name} loader yielded no batch")


def main():
    args = parse_args()
    cfg = resolve_cfg(args)
    set_seed(int(cfg["seed"]))

    print("=== Runtime check ===")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch.version.hip={torch.version.hip}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device_name={torch.cuda.get_device_name(0)}")

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda device but torch.cuda.is_available() is False")
    print(f"selected_device={device}")

    if not os.path.isdir(cfg["data_root"]):
        raise FileNotFoundError(f"data_root not found: {cfg['data_root']}")
    print(f"data_root={cfg['data_root']}")

    train_loader, val_loader = build_dataloaders(cfg)
    test_loader = build_test_loader(cfg)
    model = build_model(cfg).to(device)
    criterion = build_classification_loss()

    print("=== Pipeline smoke test ===")
    run_loader_check("train", train_loader, model, criterion, device, args.max_batches)
    run_loader_check("val", val_loader, model, criterion, device, args.max_batches)
    run_loader_check("test", test_loader, model, criterion, device, args.max_batches)
    print("All checks passed.")


if __name__ == "__main__":
    main()
