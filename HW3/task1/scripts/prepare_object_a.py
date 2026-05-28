#!/usr/bin/env python
"""Extract frames from object_A video (C1: camera orbits static object) and run COLMAP."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task1.lib.paths import (  # noqa: E402
    OBJECT_A_ROOT,
    REPO_2DGS,
    object_a_colmap_root,
    resolve_ffmpeg,
)
from task1.lib.runner import run_cmd  # noqa: E402


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    fps: float,
    max_frames: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        resolve_ffmpeg(),
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        pattern,
    ]
    run_cmd(cmd)
    return len(list(out_dir.glob("*.jpg")))


def reset_colmap_workspace(root: Path, keep_raw: bool = True) -> None:
    for name in ("input", "distorted", "sparse", "images", "stereo", "run-colmap-geometric.sh"):
        p = root / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    if not keep_raw:
        raw = root / "raw"
        if raw.is_dir():
            shutil.rmtree(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=OBJECT_A_ROOT / "raw" / "input.mp4",
        help="Orbit-around-object video (C1 workflow)",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Skip ffmpeg if you already have photos in this folder",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="COLMAP CPU mode (local Windows default)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear previous COLMAP outputs under data/object_A",
    )
    args = parser.parse_args()

    root = object_a_colmap_root()
    root.mkdir(parents=True, exist_ok=True)
    if args.reset:
        reset_colmap_workspace(root)

    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if args.images_dir:
        src = args.images_dir.resolve()
        if not src.is_dir():
            print(f"images-dir not found: {src}")
            return 1
        for f in input_dir.glob("*"):
            if f.is_file():
                f.unlink()
        count = 0
        for img in sorted(src.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                shutil.copy2(img, input_dir / f"frame_{count:04d}{img.suffix.lower()}")
                count += 1
        print(f"Copied {count} images into {input_dir}")
    else:
        if not args.video.is_file():
            print(f"Video missing: {args.video}")
            print("C1: reshoot with camera orbiting a static object, save to raw/input.mp4")
            return 1
        n = extract_frames(args.video, input_dir, fps=args.fps, max_frames=args.max_frames)
        print(f"Extracted {n} frames to {input_dir}")

    convert = REPO_2DGS / "convert.py"
    cmd = [
        sys.executable,
        str(convert),
        "-s",
        str(root),
    ]
    if args.no_gpu:
        cmd.append("--no_gpu")
    run_cmd(cmd, cwd=REPO_2DGS)

    sparse0 = root / "sparse" / "0"
    if not sparse0.is_dir():
        print("COLMAP failed: sparse/0 not created.")
        print("Common cause: turntable video (object spins, camera fixed). Use C1 orbit capture.")
        return 1

    print(f"COLMAP OK: {sparse0}")
    print("Next: DSW 上 source .venv-2dgs/bin/activate && python task1/scripts/run_train_object_a.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
