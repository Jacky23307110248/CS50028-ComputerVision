"""Subprocess helpers for calling repos/2d-gaussian-splatting and repos/threestudio."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    printable = " ".join(cmd)
    print(f"\n>>> [{cwd or Path.cwd()}] {printable}\n")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}")
    return proc


def python_exe() -> str:
    return sys.executable


def hf_offline_env() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def threestudio_launch_base(
    *,
    config: str,
    gpu: int = 0,
    extra: Iterable[str] | None = None,
) -> list[str]:
    cmd = [
        python_exe(),
        "launch.py",
        "--config",
        config,
        "--gpu",
        str(gpu),
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def find_trial_dir(exp_root: Path, name: str, tag: str) -> Path:
    trial = exp_root / name / tag
    if not trial.is_dir():
        raise FileNotFoundError(f"Expected threestudio trial dir missing: {trial}")
    return trial


def latest_ckpt(trial_dir: Path) -> Path:
    ckpt = trial_dir / "ckpts" / "last.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")
    return ckpt


def shutil_copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
