#!/usr/bin/env python
"""Train 2DGS on COLMAP-reconstructed object A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task1.lib.artifacts import snapshot_2dgs_run  # noqa: E402
from task1.lib.logging_util import RunLogger  # noqa: E402
from task1.lib.paths import OUTPUT_OBJECT_A, REPO_2DGS, ensure_output_dirs, object_a_colmap_root
from task1.lib.runner import python_exe, run_cmd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    ensure_output_dirs()
    scene = object_a_colmap_root()
    if not (scene / "sparse" / "0").is_dir():
        print("object_A COLMAP not ready. Run: python task1/scripts/prepare_object_a.py --no-gpu")
        return 1

    iters = 100 if args.smoke else args.iterations
    model_path = OUTPUT_OBJECT_A / ("smoke" if args.smoke else "run")
    model_path.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        "task1_object_a" + ("_smoke" if args.smoke else ""),
        model_path / "logs",
        use_swanlab=not args.no_swanlab,
        config={"scene": str(scene), "iterations": iters},
    )

    cmd = [
        python_exe(),
        "train.py",
        "-s",
        str(scene),
        "-m",
        str(model_path),
        "--iterations",
        str(iters),
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(args.gpu)}
    try:
        run_cmd(cmd, cwd=REPO_2DGS, env=env)
        snapshot_2dgs_run(model_path, OUTPUT_OBJECT_A)
        logger.log(iters, {"status": 1})
    finally:
        logger.finish()

    print(f"Done. Model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
