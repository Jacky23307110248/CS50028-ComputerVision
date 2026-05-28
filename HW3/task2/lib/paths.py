"""Path helpers relative to HW3 root (portable: local + /mnt/workspace/HW3)."""

from __future__ import annotations

import os
from pathlib import Path

# task2/lib/paths.py -> HW3 root
HW3_ROOT = Path(__file__).resolve().parents[2]

# Override on AMD if needed: export HW3_ROOT=/mnt/workspace/HW3
if env_root := os.environ.get("HW3_ROOT"):
    HW3_ROOT = Path(env_root).resolve()

DATA_ROOT = HW3_ROOT / "data" / "calvin_task_ABC_D"
LEROBOT_SRC = HW3_ROOT / "lerobot" / "src"

ENV_SHARDS = {
    "A": "calvin_task_ABC_D_lerobot_0_4",
    "B": "calvin_task_ABC_D_lerobot_1_4",
    "C": "calvin_task_ABC_D_lerobot_2_4",
    "D": "calvin_task_ABC_D_lerobot_3_4",
}

TASK2_ROOT = HW3_ROOT / "task2"
CONFIGS_DIR = TASK2_ROOT / "configs"
SPLITS_PATH = CONFIGS_DIR / "splits.json"
OUTPUTS_DIR = TASK2_ROOT / "outputs"


def env_dataset_root(env: str) -> Path:
    if env not in ENV_SHARDS:
        raise KeyError(f"Unknown env {env!r}, expected one of {list(ENV_SHARDS)}")
    return DATA_ROOT / ENV_SHARDS[env]


def dataset_codebase_version(env: str) -> str:
    import json

    info_path = env_dataset_root(env) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return str(info.get("codebase_version", ""))


def ensure_lerobot_on_path() -> None:
    import sys

    src = str(LEROBOT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
