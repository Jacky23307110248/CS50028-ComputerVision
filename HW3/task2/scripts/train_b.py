#!/usr/bin/env python
"""Experiment 1: ACT trained on environment B train split only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task2.lib.datasets import make_b_train_val_datasets  # noqa: E402
from task2.lib.splits import load_splits  # noqa: E402
from task2.lib.train_loop import TrainHyperparams, train_act  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Short run for testing")
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()

    splits = load_splits()
    hparams = TrainHyperparams()
    if args.smoke:
        hparams.steps = 20
        hparams.log_freq = 5
        hparams.eval_freq = 10
        hparams.save_freq = 20
        hparams.batch_size = 2

    train_ds, val_ds = make_b_train_val_datasets(splits)

    print(
        f"B train episodes: {len(splits.train_episodes('B'))}, "
        f"B val episodes: {len(splits.val_episodes('B'))}"
    )
    print("Environment D is NOT used (test-only).")

    out = train_act(
        run_name="act_B_only" + ("_smoke" if args.smoke else ""),
        train_dataset=train_ds,
        val_dataset=val_ds,
        hparams=hparams,
        use_swanlab=not args.no_swanlab,
    )
    print(f"Done. Outputs: {out}")


if __name__ == "__main__":
    main()
