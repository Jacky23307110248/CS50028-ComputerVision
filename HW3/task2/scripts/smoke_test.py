#!/usr/bin/env python
"""End-to-end smoke: build splits (if needed), short B train, eval on D (few batches)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=HW3_ROOT)


def main() -> None:
    splits = HW3_ROOT / "task2" / "configs" / "splits.json"
    if not splits.is_file():
        run([sys.executable, str(SCRIPTS / "build_splits.py")])

    run([sys.executable, str(SCRIPTS / "check_setup.py")])

    run(
        [
            sys.executable,
            str(SCRIPTS / "train_b.py"),
            "--smoke",
            "--no-swanlab",
        ]
    )

    last_ckpt = HW3_ROOT / "task2" / "outputs" / "act_B_only_smoke" / "last_checkpoint.txt"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint written: {last_ckpt}")
    ckpt = Path(last_ckpt.read_text(encoding="utf-8").strip())

    run(
        [
            sys.executable,
            str(SCRIPTS / "eval_zero_shot_d.py"),
            "--checkpoint",
            str(ckpt),
            "--run-name",
            "smoke_eval_D",
            "--smoke",
        ]
    )
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
