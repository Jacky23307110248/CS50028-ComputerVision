"""从视频中按帧号导出 PNG，便于报告截取连续帧。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402

from src.paths import DEFAULT_INPUT_VIDEO  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="按帧号导出视频帧为 PNG")
    p.add_argument("--video", type=Path, default=DEFAULT_INPUT_VIDEO)
    p.add_argument("--frames", type=int, nargs="+", required=True, help="0-based 帧号列表")
    p.add_argument("--outdir", type=Path, default=_ROOT / "detectVideo" / "frames_for_report")
    args = p.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"找不到视频: {video}")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    for fi in sorted(set(args.frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[warn] 无法读取帧 {fi}")
            continue
        out_path = outdir / f"frame_{fi:06d}.png"
        cv2.imwrite(str(out_path), frame)
        print(out_path)
    cap.release()


if __name__ == "__main__":
    main()
