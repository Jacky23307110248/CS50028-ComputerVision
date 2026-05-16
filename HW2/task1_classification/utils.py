import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def prepare_experiment_dir(output_root: str, exp_name: str) -> str:
    ensure_dir(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_root, f"{exp_name}_{timestamp}")
    ensure_dir(exp_dir)
    ensure_dir(os.path.join(exp_dir, "figs"))
    return exp_dir

