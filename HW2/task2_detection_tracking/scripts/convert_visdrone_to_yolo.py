"""
将 VisDrone 原生 txt 标注转为 YOLO 格式，写入与 images 同级的 labels/ 目录。
不复制图片；跳过 score==0 及类别不在 1..10 的框。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image  # noqa: E402

from src.paths import EVILSPIRIT14  # noqa: E402


def _read_image_wh(img_path: Path, ann_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(img_path) as pil_im:
            return pil_im.size  # (w, h)
    except OSError:
        pass
    mw, mh = 0, 0
    if ann_path.is_file():
        for raw in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.strip().split(",")
            if len(parts) < 6:
                continue
            bx, by, bw, bh = map(float, parts[:4])
            mw = max(mw, int(bx + bw + 0.999))
            mh = max(mh, int(by + bh + 0.999))
    if mw <= 0 or mh <= 0:
        return None
    return mw, mh


def _convert_one_image(img_path: Path, ann_path: Path, out_label: Path) -> int:
    wh = _read_image_wh(img_path, ann_path)
    if wh is None:
        return 0
    w, h = wh
    lines_out: list[str] = []
    if ann_path.is_file():
        text = ann_path.read_text(encoding="utf-8", errors="ignore").strip()
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                continue
            bx, by, bw, bh = map(float, parts[:4])
            score = int(float(parts[4]))
            cat = int(float(parts[5]))
            if score == 0:
                continue
            if cat < 1 or cat > 10:
                continue
            yolo_c = cat - 1
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 1e-6), 1.0)
            nh = min(max(nh, 1e-6), 1.0)
            lines_out.append(f"{yolo_c} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    out_label.parent.mkdir(parents=True, exist_ok=True)
    out_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return len(lines_out)


def main() -> None:
    p = argparse.ArgumentParser(description="VisDrone txt -> YOLO labels（写入 labels/，不复制图片）")
    p.add_argument(
        "--root",
        type=Path,
        default=EVILSPIRIT14,
        help="visdrone versions/14 根目录",
    )
    args = p.parse_args()
    root = args.root.resolve()

    splits = (
        (
            "train",
            root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train" / "images",
            root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train" / "annotations",
        ),
        (
            "val",
            root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "images",
            root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "annotations",
        ),
    )

    total_files = 0
    total_boxes = 0
    for split_name, images_dir, ann_dir in splits:
        if not images_dir.is_dir():
            print(f"[skip] 无目录: {images_dir}")
            continue
        labels_dir = images_dir.parent / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        jpgs = sorted(images_dir.glob("*.jpg"))
        print(f"[{split_name}] 图像数: {len(jpgs)}, 标注目录: {ann_dir}")
        for img_path in jpgs:
            ann_path = ann_dir / f"{img_path.stem}.txt"
            out_label = labels_dir / f"{img_path.stem}.txt"
            n = _convert_one_image(img_path, ann_path, out_label)
            total_boxes += n
            total_files += 1
        print(f"[{split_name}] 已写入 labels: {labels_dir}")

    print(f"完成: 处理图像 {total_files} 张, 总框数约 {total_boxes}")


if __name__ == "__main__":
    main()
