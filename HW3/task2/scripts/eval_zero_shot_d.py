#!/usr/bin/env python
"""Offline eval: D zero-shot (d_test), B val (b_val), or ABC val (abc_val)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task2.lib.eval_d import eval_offline  # noqa: E402
from task2.lib.paths import OUTPUTS_DIR  # noqa: E402
from task2.lib.splits import load_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HW3 task2 offline evaluation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="pretrained_model dir (e.g. .../checkpoints/0100000/pretrained_model)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="d_test",
        choices=["d_test", "b_val", "abc_val"],
        help="d_test=env D; b_val=B-only ID; abc_val=A+B+C val ID",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Limit batches (debug only)")
    args = parser.parse_args()

    splits = load_splits()
    if args.split == "d_test":
        n_ep = len(splits.test_episodes("D"))
    elif args.split == "b_val":
        n_ep = len(splits.val_episodes("B"))
    else:
        n_ep = sum(len(splits.val_episodes(e)) for e in ("A", "B", "C"))
    print(f"Split {args.split}: {n_ep} episodes")

    run_name = args.run_name or f"eval_{args.split}"
    out_dir = OUTPUTS_DIR / run_name
    summary = eval_offline(
        checkpoint_dir=args.checkpoint,
        output_dir=out_dir,
        split=args.split,
        max_batches=3 if args.smoke else None,
        model_tag=args.model_tag,
    )
    print(f"Done. l1_norm={summary['l1_norm']:.6f} mae_denorm={summary['mae_denorm']:.6f}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
