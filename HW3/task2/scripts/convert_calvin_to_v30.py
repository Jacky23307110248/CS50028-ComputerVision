#!/usr/bin/env python
"""
One-time in-place conversion: CALVIN shards v2.1 → LeRobot v3.0.
Required before training/eval with current lerobot. Does not touch test/train logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task2.lib.paths import env_dataset_root  # noqa: E402
from task2.lib.splits import ALL_ENVS  # noqa: E402

CODEBASE_V30 = "v3.0"


def _version(env: str) -> str:
    info = json.loads((env_dataset_root(env) / "meta" / "info.json").read_text(encoding="utf-8"))
    return str(info.get("codebase_version", ""))


def convert_env(env: str, force: bool) -> None:
    root = env_dataset_root(env)
    if not force and _version(env) == CODEBASE_V30:
        print(f"  {env}: already {CODEBASE_V30}, skip")
        return
    print(f"  {env}: converting {root} ...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.convert_dataset_v21_to_v30",
            f"--repo-id=local/calvin_{env}",
            f"--root={root}",
            "--push-to-hub=false",
            *(["--force-conversion"] if force else []),
        ],
        cwd=HW3_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help="Which shard to convert (default: all four)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    envs = list(ALL_ENVS) if args.env == "all" else [args.env]
    for env in envs:
        convert_env(env, args.force)
    print("Conversion finished.")


if __name__ == "__main__":
    main()
