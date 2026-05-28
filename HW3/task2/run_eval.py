#!/usr/bin/env python
"""
Short eval launcher (defaults for AMD). From HW3 root:

  python task2/run_eval.py 1    # B-only  @ D test
  python task2/run_eval.py 2    # ABC     @ D test
  python task2/run_eval.py 3    # B-only  @ B val (ID)
  python task2/run_eval.py 4    # ABC     @ ABC val (ID)
  python task2/run_eval.py m    # merge CSV from 1-4

Progress: tail -f task2/outputs/<run>/eval.log
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_ROOT))

from task2.lib.eval_d import eval_offline  # noqa: E402
from task2.lib.paths import OUTPUTS_DIR  # noqa: E402

CKPT_B = OUTPUTS_DIR / "act_B_only/checkpoints/0100000/pretrained_model"
CKPT_ABC = OUTPUTS_DIR / "act_ABC_mixed/checkpoints/0100000/pretrained_model"

JOBS: dict[str, dict] = {
    "1": {
        "split": "d_test",
        "checkpoint": CKPT_B,
        "run_name": "eval_D_B_only",
        "model_tag": "act_B_only",
    },
    "2": {
        "split": "d_test",
        "checkpoint": CKPT_ABC,
        "run_name": "eval_D_ABC",
        "model_tag": "act_ABC_mixed",
    },
    "3": {
        "split": "b_val",
        "checkpoint": CKPT_B,
        "run_name": "eval_B_val_B_only",
        "model_tag": "act_B_only",
    },
    "4": {
        "split": "abc_val",
        "checkpoint": CKPT_ABC,
        "run_name": "eval_ABC_val_ABC",
        "model_tag": "act_ABC_mixed",
    },
}


def _run_job(key: str) -> None:
    job = JOBS[key]
    ckpt = Path(job["checkpoint"])
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    out = OUTPUTS_DIR / job["run_name"]
    print(f">>> [{key}] {job['model_tag']} split={job['split']}")
    print(f"    ckpt={ckpt}")
    print(f"    out={out}")
    summary = eval_offline(
        checkpoint_dir=ckpt,
        output_dir=out,
        split=job["split"],
        model_tag=job["model_tag"],
    )
    print(f"    l1_norm={summary['l1_norm']:.6f}  log={out / 'eval.log'}")


def _merge() -> None:
    script = HW3_ROOT / "task2/scripts/merge_eval_summaries.py"
    out = OUTPUTS_DIR / "eval_compare_id_ood.csv"
    specs = [
        f"B_D={OUTPUTS_DIR / 'eval_D_B_only'}",
        f"ABC_D={OUTPUTS_DIR / 'eval_D_ABC'}",
        f"B_val={OUTPUTS_DIR / 'eval_B_val_B_only'}",
        f"ABC_val={OUTPUTS_DIR / 'eval_ABC_val_ABC'}",
    ]
    subprocess.check_call(
        [sys.executable, str(script), "--runs", *specs, "--out", str(out)],
        cwd=HW3_ROOT,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("Jobs:", ", ".join(f"{k}={v['run_name']}" for k, v in JOBS.items()))
        return
    key = sys.argv[1].strip().lower()
    if key in ("m", "merge", "5"):
        _merge()
        return
    if key not in JOBS:
        raise SystemExit(f"Unknown job {key!r}. Use 1, 2, 3, 4, or m.")
    _run_job(key)


if __name__ == "__main__":
    main()
