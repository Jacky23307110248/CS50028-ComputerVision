"""在 VisDrone 上微调 YOLOv8；数据配置见 configs/visdrone_local.yaml（需先运行 convert_visdrone_to_yolo.py）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO  # noqa: E402

from src.paths import VISDRONE_DATA_YAML  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="YOLOv8 VisDrone 微调")
    p.add_argument("--data", type=Path, default=VISDRONE_DATA_YAML, help="data yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="预训练起点，如 yolov8n.pt / yolov8s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="", help="留空自动；可填 0 或 cpu")
    p.add_argument("--project", type=Path, default=_ROOT / "runs" / "detect", help="Ultralytics 输出根目录")
    p.add_argument("--name", type=str, default="visdrone", help="本次实验子目录名")
    args = p.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"找不到 data yaml: {data_path}")

    model = YOLO(args.model)
    train_kw = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project.resolve()),
        name=args.name,
    )
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)
    print(f"训练结束。权重通常在: {args.project.resolve() / args.name / 'weights' / 'best.pt'}")
    print("可将其复制到 task2_detection_tracking/weights/best.pt 供 infer_track 使用。")


if __name__ == "__main__":
    main()
