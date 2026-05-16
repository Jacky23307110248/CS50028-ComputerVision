import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_run_name(cfg: dict) -> str:
    if cfg.get("run_name"):
        return cfg["run_name"]
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep run names compact but make them sensitive to key CLI hyper-parameters.
    return (
        f"{now}_unet_"
        f"{cfg['loss']}_"
        f"ep{cfg['epochs']}_"
        f"bs{cfg['batch_size']}_"
        f"img{cfg['img_size']}_"
        f"lr{cfg['lr']}_"
        f"wd{cfg['weight_decay']}_"
        f"cw{cfg['ce_weight']}_"
        f"dw{cfg['dice_weight']}_"
        f"seed{cfg['seed']}"
    )


def prepare_run_dir(cfg: dict) -> Path:
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(cfg)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def init_logger(run_dir: Path):
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def save_config(run_dir: Path, cfg: dict):
    with open(run_dir / "config_merged.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def init_metrics_csv(run_dir: Path):
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_miou", "val_loss", "val_miou"])


def append_metrics_csv(run_dir: Path, epoch: int, train_stats: dict, val_stats: dict):
    with open(run_dir / "metrics.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                f"{train_stats['loss']:.6f}",
                f"{train_stats['miou']:.6f}",
                f"{val_stats['loss']:.6f}",
                f"{val_stats['miou']:.6f}",
            ]
        )


def save_result(run_dir: Path, result: dict):
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
