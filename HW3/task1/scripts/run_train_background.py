#!/usr/bin/env python
"""Train 2DGS on Mip-NeRF 360 garden background."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task1.lib.artifacts import snapshot_2dgs_run  # noqa: E402
from task1.lib.logging_util import RunLogger  # noqa: E402
from task1.lib.paths import (  # noqa: E402
    OUTPUT_BACKGROUND,
    REPO_2DGS,
    ensure_output_dirs,
    garden_scene_root,
)
from task1.lib.runner import python_exe, run_cmd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--smoke", action="store_true", help="Short run for pipeline test")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index for 2DGS")
    args = parser.parse_args()

    ensure_output_dirs()
    scene = garden_scene_root()
    if not (scene / "images").is_dir():
        print(f"garden scene missing: {scene}")
        return 1

    iters = 100 if args.smoke else args.iterations
    model_path = OUTPUT_BACKGROUND / ("smoke" if args.smoke else "run")
    model_path.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        "task1_background_garden" + ("_smoke" if args.smoke else ""),
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
        snapshot_2dgs_run(model_path, OUTPUT_BACKGROUND)
        logger.log(iters, {"status": 1, "model_path": str(model_path)})
    finally:
        logger.finish()

    print(f"Done. Model: {model_path}")
    print("Artifacts: task1/outputs/background_garden/artifacts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
