#!/usr/bin/env python
"""Object C: Magic123 coarse + refine (SD 1.5 + Zero123) from single RGBA image + prompt."""

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
from task1.lib.paths import OUTPUT_FUSION, OUTPUT_OBJECT_C, OBJECT_C_ROOT, ensure_output_dirs
from task1.lib.prompts import PromptFiles  # noqa: E402
from task1.lib.runner import latest_ckpt  # noqa: E402
from task1.lib.threestudio_cli import (  # noqa: E402
    export_mesh,
    magic123_coarse_overrides,
    magic123_refine_overrides,
    train_threestudio,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=OBJECT_C_ROOT / "rgba.png",
        help="RGBA foreground image (alpha = mask)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--coarse-steps", type=int, default=2000)
    parser.add_argument("--refine-steps", type=int, default=2000)
    args = parser.parse_args()

    ensure_output_dirs()
    if not args.image.is_file():
        print(f"Missing image: {args.image}")
        print("Place a cut-out RGBA PNG at data/object_C/rgba.png")
        return 1

    pf = PromptFiles(OBJECT_C_ROOT)
    prompt = pf.require_prompt()
    negative = pf.negative_prompt()

    coarse_steps = 30 if args.smoke else args.coarse_steps
    refine_steps = 30 if args.smoke else args.refine_steps

    logger = RunLogger(
        "task1_object_c_magic123" + ("_smoke" if args.smoke else ""),
        OUTPUT_OBJECT_C / "logs",
        use_swanlab=not args.no_swanlab,
        config={
            "prompt": prompt,
            "image": str(args.image),
            "coarse_steps": coarse_steps,
            "refine_steps": refine_steps,
        },
    )

    try:
        coarse_trial = train_threestudio(
            config="configs/magic123-coarse-sd.yaml",
            name="object_c",
            tag="coarse",
            overrides=magic123_coarse_overrides(
                image_path=args.image.resolve(),
                prompt=prompt,
                negative_prompt=negative,
                max_steps=coarse_steps,
            ),
            gpu=args.gpu,
        )
        snapshot_threestudio_trial(
            coarse_trial,
            OUTPUT_OBJECT_C,
            stage="coarse",
            prompt=prompt,
            negative_prompt=negative,
        )
        logger.log(coarse_steps, {"stage": "coarse_done"})

        final_trial = coarse_trial
        if not args.skip_refine:
            coarse_ckpt = latest_ckpt(coarse_trial)
            refine_trial = train_threestudio(
                config="configs/magic123-refine-sd.yaml",
                name="object_c",
                tag="refine",
                overrides=magic123_refine_overrides(
                    image_path=args.image.resolve(),
                    prompt=prompt,
                    negative_prompt=negative,
                    coarse_ckpt=coarse_ckpt,
                    max_steps=refine_steps,
                ),
                gpu=args.gpu,
            )
            snapshot_threestudio_trial(
                refine_trial,
                OUTPUT_OBJECT_C,
                stage="refine",
                prompt=prompt,
                negative_prompt=negative,
            )
            final_trial = refine_trial
            logger.log(coarse_steps + refine_steps, {"stage": "refine_done"})

        if not args.no_export:
            export_mesh(final_trial, gpu=args.gpu)
            copy_export_meshes(final_trial, OUTPUT_FUSION / "blender", "object_c")

        logger.log(coarse_steps + refine_steps, {"status": 1, "trial": str(final_trial)})
    finally:
        logger.finish()

    print(f"Done. Final trial: {final_trial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
