#!/usr/bin/env python
"""Generate train/val/test episode splits. D -> test only."""

from __future__ import annotations

import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task2.lib.splits import build_splits, load_splits, save_splits  # noqa: E402


def main() -> None:
    spec = build_splits(seed=42, val_ratio=0.05)
    path = save_splits(spec)
    spec2 = load_splits(path)
    spec2.assert_no_leakage()
    print(f"Saved splits to {path}")
    for env in ("A", "B", "C"):
        print(
            f"  {env}: train={len(spec2.train[env])} val={len(spec2.val[env])} "
            f"(total={len(spec2.train[env]) + len(spec2.val[env])})"
        )
    print(f"  D: test={len(spec2.test['D'])} (all episodes, never used in train/val)")


if __name__ == "__main__":
    main()
