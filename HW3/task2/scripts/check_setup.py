#!/usr/bin/env python
"""Verify data, splits, lerobot import, and that D is excluded from train paths."""

from __future__ import annotations

import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task2.lib.paths import (  # noqa: E402
    DATA_ROOT,
    ENV_SHARDS,
    HW3_ROOT,
    LEROBOT_SRC,
    SPLITS_PATH,
    dataset_codebase_version,
)
from task2.lib.splits import ALL_ENVS, TEST_ENV, TRAIN_ENVS, load_splits  # noqa: E402

REQUIRED_DATASET_VERSION = "v3.0"


def main() -> int:
    print(f"HW3_ROOT={HW3_ROOT}")
    print(f"LEROBOT_SRC exists: {LEROBOT_SRC.is_dir()}")

    ok = True
    for env, shard in ENV_SHARDS.items():
        root = DATA_ROOT / shard
        exists = root.is_dir()
        print(f"  env {env}: {root} -> {'OK' if exists else 'MISSING'}")
        ok = ok and exists

    if not SPLITS_PATH.is_file():
        print(f"MISSING {SPLITS_PATH} — run: python task2/scripts/build_splits.py")
        return 1

    splits = load_splits()
    splits.assert_no_leakage()
    print("splits.json: leakage checks passed")

    for env in TRAIN_ENVS:
        print(
            f"  {env}: train={len(splits.train_episodes(env))} "
            f"val={len(splits.val_episodes(env))}"
        )
    print(f"  {TEST_ENV}: test={len(splits.test_episodes(TEST_ENV))} (never in train/val)")

    for env in ALL_ENVS:
        ver = dataset_codebase_version(env)
        if ver != REQUIRED_DATASET_VERSION:
            print(
                f"  env {env}: codebase_version={ver} (need {REQUIRED_DATASET_VERSION}) "
                f"→ run: python task2/scripts/convert_calvin_to_v30.py --env {env}"
            )
            ok = False
        else:
            print(f"  env {env}: {REQUIRED_DATASET_VERSION} OK")

    try:
        from task2.lib.paths import ensure_lerobot_on_path

        ensure_lerobot_on_path()
        import lerobot  # noqa: F401

        print(f"lerobot import: OK ({lerobot.__file__})")
    except Exception as exc:  # noqa: BLE001
        print(f"lerobot import FAILED: {exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
