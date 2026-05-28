#!/usr/bin/env python
"""Object B: DreamFusion (SDS + SD 2.1) from text prompt only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task1.lib.artifacts import (  # noqa: E402
    copy_export_meshes,
    snapshot_threestudio_trial,
)
from task1.lib.logging_util import RunLogger  # noqa: E402
from task1.lib.paths import OUTPUT_FUSION, OUTPUT_OBJECT_B, OBJECT_B_ROOT, ensure_output_dirs
from task1.lib.prompts import PromptFiles  # noqa: E402
from task1.lib.threestudio_cli import dreamfusion_overrides, export_mesh, train_threestudio  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()

    ensure_output_dirs()
    pf = PromptFiles(OBJECT_B_ROOT)
    prompt = pf.require_prompt()
    negative = pf.negative_prompt()
    max_steps = 50 if args.smoke else args.max_steps

    logger = RunLogger(
        "task1_object_b_dreamfusion" + ("_smoke" if args.smoke else ""),
        OUTPUT_OBJECT_B / "logs",
        use_swanlab=not args.no_swanlab,
        config={"prompt": prompt, "max_steps": max_steps},
    )

    try:
        trial = train_threestudio(
            config="configs/dreamfusion-sd.yaml",
            name="object_b",
            tag="dreamfusion",
            overrides=dreamfusion_overrides(
                prompt=prompt,
                negative_prompt=negative,
                max_steps=max_steps,
            ),
            gpu=args.gpu,
        )
        snapshot_threestudio_trial(
            trial,
            OUTPUT_OBJECT_B,
            stage="dreamfusion",
            prompt=prompt,
            negative_prompt=negative,
        )
        if not args.no_export:
            export_mesh(trial, gpu=args.gpu)
            copy_export_meshes(trial, OUTPUT_FUSION / "blender", "object_b")
        logger.log(max_steps, {"status": 1, "trial": str(trial)})
    finally:
        logger.finish()

    print(f"Done. Trial: {trial}")
    print("Blender meshes (if exported): task1/outputs/fusion/blender/object_b_*.obj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
