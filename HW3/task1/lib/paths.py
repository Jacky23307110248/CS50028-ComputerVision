"""Path helpers relative to HW3 root (local + /mnt/workspace/HW3)."""

from __future__ import annotations

import os
from pathlib import Path

# task1/lib/paths.py -> HW3 root
HW3_ROOT = Path(__file__).resolve().parents[2]

if env_root := os.environ.get("HW3_ROOT"):
    HW3_ROOT = Path(env_root).resolve()

TASK1_ROOT = HW3_ROOT / "task1"
CONFIGS_DIR = TASK1_ROOT / "configs"
OUTPUTS_DIR = TASK1_ROOT / "outputs"

REPO_2DGS = HW3_ROOT / "repos" / "2d-gaussian-splatting"
REPO_THREESTUDIO = HW3_ROOT / "repos" / "threestudio"

CACHE_HF = HW3_ROOT / "cache" / "hf"
SD15_PATH = CACHE_HF / "stable-diffusion-v1-5"
SD21_PATH = CACHE_HF / "stable-diffusion-2-1-base"
ZERO123_PATH = CACHE_HF / "zero123-diffusers"

DATA = HW3_ROOT / "data"
OBJECT_A_ROOT = DATA / "object_A"
OBJECT_B_ROOT = DATA / "object_B"
OBJECT_C_ROOT = DATA / "object_C"
MIPNERF360_ROOT = DATA / "mipnerf360"

OUTPUT_BACKGROUND = OUTPUTS_DIR / "background_garden"
OUTPUT_OBJECT_A = OUTPUTS_DIR / "object_a"
OUTPUT_OBJECT_B = OUTPUTS_DIR / "object_b"
OUTPUT_OBJECT_C = OUTPUTS_DIR / "object_c"
OUTPUT_FUSION = OUTPUTS_DIR / "fusion"
THREESTUDIO_EXP_ROOT = OUTPUTS_DIR / "threestudio"

# DSW 无 conda：两套 venv，继承系统预装 torch/cuda
VENV_2DGS = HW3_ROOT / ".venv-2dgs"
VENV_THREESTUDIO = HW3_ROOT / ".venv-ts"

SWANLAB_PROJECT = "CVHW3"
SWANLAB_WORKSPACE = os.environ.get("SWANLAB_WORKSPACE", "23307110248JackyH")


def garden_scene_root() -> Path:
    """Return Mip-NeRF 360 garden COLMAP scene root (images/ + sparse/)."""
    candidates = [
        MIPNERF360_ROOT / "garden",
        MIPNERF360_ROOT / "_download" / "360_v2" / "garden",
    ]
    for root in candidates:
        if (root / "images").is_dir() and (
            (root / "sparse").is_dir() or (root / "poses_bounds.npy").is_file()
        ):
            return root
    return candidates[0]


def object_a_colmap_root() -> Path:
    """COLMAP + 2DGS source tree for object A (input/ + sparse/)."""
    return OBJECT_A_ROOT


def resolve_ffmpeg() -> str:
    if exe := os.environ.get("FFMPEG"):
        return exe
    return "ffmpeg"


def venv_python(which: str) -> Path:
    """Return python executable inside HW3 venv ('2dgs' | 'ts')."""
    root = VENV_2DGS if which == "2dgs" else VENV_THREESTUDIO
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def ensure_output_dirs() -> None:
    for d in (
        OUTPUT_BACKGROUND,
        OUTPUT_OBJECT_A,
        OUTPUT_OBJECT_B,
        OUTPUT_OBJECT_C,
        OUTPUT_FUSION,
        THREESTUDIO_EXP_ROOT,
        OUTPUT_FUSION / "export",
        OUTPUT_FUSION / "blender",
    ):
        d.mkdir(parents=True, exist_ok=True)
