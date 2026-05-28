#!/usr/bin/env python
"""Experiment 2: ACT trained on A+B+C train splits (ConcatDataset). Same hyperparams as B-only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task2.lib.datasets import make_abc_train_val_datasets  # noqa: E402
from task2.lib.splits import TRAIN_ENVS, load_splits  # noqa: E402
from task2.lib.train_loop import TrainHyperparams, train_act  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
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

    train_ds, val_ds = make_abc_train_val_datasets(splits)

    for env in TRAIN_ENVS:
        print(
            f"{env} train={len(splits.train_episodes(env))} "
            f"val={len(splits.val_episodes(env))}"
        )
    print("Environment D is NOT used (test-only).")

    out = train_act(
        run_name="act_ABC_mixed" + ("_smoke" if args.smoke else ""),
        train_dataset=train_ds,
        val_dataset=val_ds,
        hparams=hparams,
        use_swanlab=not args.no_swanlab,
    )
    print(f"Done. Outputs: {out}")


if __name__ == "__main__":
    main()
