"""
自视频：检测 + 多目标跟踪，可选虚拟越线计数（须传入 x1 y1 x2 y2）。
输出 output.mp4，并写 track_log.csv（每帧各 track 的框中心与类别等）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Windows：须在 import cv2 之前设置，减轻 GStreamer 误解析 D:\ 路径等问题
if sys.platform == "win32":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "MSMF,FFMPEG,GSTREAMER")

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO  # noqa: E402

from src.line_counter import LineCounter, parse_line_arg  # noqa: E402
from src.paths import (  # noqa: E402
    DEFAULT_INPUT_VIDEO,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_TRACK_LOG,
    DEFAULT_WEIGHTS,
)


def _path_has_nonascii(s: str) -> bool:
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


def _probe_video_size(path: Path) -> Tuple[float, int, int]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-3:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return float(fps), w, h
    finally:
        cap.release()


def _ascii_temp_video_copy(src: Path) -> Path:
    fd, name = tempfile.mkstemp(suffix=".mp4", prefix="infer_track_src_")
    os.close(fd)
    dst = Path(name)
    shutil.copy2(src, dst)
    return dst


def _draw_line_and_text(
    img: np.ndarray,
    line: Optional[Tuple[float, float, float, float]],
    count: Optional[int],
) -> None:
    if line is not None:
        h, w = img.shape[:2]
        ok, p1, p2 = cv2.clipLine((0, 0, w, h), (int(line[0]), int(line[1])), (int(line[2]), int(line[3])))
        if ok:
            cv2.line(img, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
    if count is not None:
        cv2.putText(
            img,
            f"line_crossings={count}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _line_dict(line: Optional[Tuple[float, float, float, float]]) -> Optional[dict]:
    if line is None:
        return None
    return {"x1": float(line[0]), "y1": float(line[1]), "x2": float(line[2]), "y2": float(line[3])}


def main() -> None:
    p = argparse.ArgumentParser(description="视频检测 + 跟踪 + 可选越线计数")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--source", type=Path, default=DEFAULT_INPUT_VIDEO)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    p.add_argument("--csv", type=Path, default=DEFAULT_TRACK_LOG)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker 配置名")
    p.add_argument("--device", type=str, default="", help="留空自动")
    p.add_argument(
        "--line",
        type=float,
        nargs="*",
        default=None,
        help="四个数 x1 y1 x2 y2（像素，须与视频分辨率一致、线段宜横穿车流）；不传则只跟踪、不画线、不计数",
    )
    args = p.parse_args()

    src = args.source.resolve()
    if not src.is_file():
        raise SystemExit(f"找不到输入视频: {src}")

    wts = args.weights.resolve()
    if not wts.is_file():
        raise SystemExit(f"找不到权重: {wts}（请先训练并复制 best.pt 到 weights/）")

    line_tuple = parse_line_arg(args.line if args.line else None)
    counter: Optional[LineCounter] = None
    if line_tuple is not None:
        counter = LineCounter(*line_tuple)

    track_src = src
    temp_src: Optional[Path] = None
    if sys.platform == "win32" and _path_has_nonascii(str(src)):
        temp_src = _ascii_temp_video_copy(src)
        track_src = temp_src
    fps, w, h = _probe_video_size(track_src)
    if (w <= 0 or h <= 0) and sys.platform == "win32" and temp_src is None:
        temp_src = _ascii_temp_video_copy(src)
        track_src = temp_src
        fps, w, h = _probe_video_size(track_src)
    if w <= 0 or h <= 0:
        if temp_src is not None and temp_src.is_file():
            try:
                temp_src.unlink()
            except OSError:
                pass
        raise SystemExit(
            f"无法读取视频分辨率（{w}x{h}）。常见原因：路径含中文/盘符时 OpenCV 打不开文件。\n"
            f"  已尝试临时副本仍失败请检查文件是否损坏，或将视频复制到纯英文路径后用 --source 指定。\n"
            f"  路径: {src}"
        )

    if line_tuple is not None:
        ok_clip, _, _ = cv2.clipLine(
            (0, 0, w, h),
            (int(line_tuple[0]), int(line_tuple[1])),
            (int(line_tuple[2]), int(line_tuple[3])),
        )
        if not ok_clip:
            print(
                f"警告: --line 与画面 {w}x{h} 无交集，黄线不会显示；"
                f"且计数依据的是「无限延长直线」，若该线不穿过目标活动区域，计数会恒为 0。\n"
                f"  当前: --line {line_tuple[0]} {line_tuple[1]} {line_tuple[2]} {line_tuple[3]}\n"
                f"  请把线段画在画面内并横穿车流（例如横线 y 取 0~{h - 1} 之间）。",
                file=sys.stderr,
            )

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    write_path = out_path
    temp_out: Optional[Path] = None
    if sys.platform == "win32" and _path_has_nonascii(str(out_path)):
        fd_o, name_o = tempfile.mkstemp(suffix=".mp4", prefix="infer_track_out_")
        os.close(fd_o)
        temp_out = Path(name_o)
        write_path = temp_out
    writer = cv2.VideoWriter(str(write_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        if temp_out is not None and temp_out.is_file():
            try:
                temp_out.unlink()
            except OSError:
                pass
            temp_out = None
        if sys.platform == "win32":
            fd_o, name_o = tempfile.mkstemp(suffix=".mp4", prefix="infer_track_out_")
            os.close(fd_o)
            temp_out = Path(name_o)
            write_path = temp_out
            writer = cv2.VideoWriter(str(write_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        if temp_src is not None and temp_src.is_file():
            try:
                temp_src.unlink()
            except OSError:
                pass
        raise SystemExit(f"无法创建输出视频。目标: {out_path}")

    csv_path = args.csv.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = csv_path.parent / "track_run_meta.json"
    f_csv = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(f_csv)
    csv_w.writerow(["frame", "track_id", "cx", "cy", "cls", "conf", "x1", "y1", "x2", "y2"])

    model = YOLO(str(wts))
    track_kw = dict(
        source=str(track_src),
        stream=True,
        imgsz=args.imgsz,
        conf=0.25,
        iou=0.45,
        persist=True,
        verbose=False,
        tracker=args.tracker,
    )
    if args.device:
        track_kw["device"] = args.device

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(src),
        "output_video": str(out_path),
        "track_log_csv": str(csv_path),
        "weights": str(wts),
        "tracker": args.tracker,
        "imgsz": int(args.imgsz),
        "line_params": _line_dict(line_tuple),
        "line_counting_enabled": line_tuple is not None,
        "video_width": int(w),
        "video_height": int(h),
        "video_fps": float(fps),
        "nms_conf_threshold": 0.25,
        "nms_iou_threshold": 0.45,
    }
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(run_meta, f_meta, ensure_ascii=False, indent=2)

    frame_idx = 0
    try:
        for result in model.track(**track_kw):
            plot_img = result.plot()
            if plot_img.shape[1] != w or plot_img.shape[0] != h:
                plot_img = cv2.resize(plot_img, (w, h))

            ids_list: List[int] = []
            centers: List[Tuple[float, float]] = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                tids = result.boxes.id
                if tids is None:
                    tids_np = np.full(len(xyxy), -1, dtype=np.int64)
                else:
                    tids_np = tids.int().cpu().numpy()

                for i in range(len(xyxy)):
                    x1b, y1b, x2b, y2b = xyxy[i]
                    tid = int(tids_np[i])
                    c = float(cls[i])
                    cf = float(conf[i])
                    cx = float((x1b + x2b) / 2.0)
                    cy = float((y1b + y2b) / 2.0)
                    if tid >= 0:
                        ids_list.append(tid)
                        centers.append((cx, cy))
                    csv_w.writerow(
                        [frame_idx, tid, f"{cx:.2f}", f"{cy:.2f}", int(c), f"{cf:.4f}", int(x1b), int(y1b), int(x2b), int(y2b)]
                    )

            count_disp: Optional[int] = None
            if counter is not None:
                counter.update(ids_list, centers)
                count_disp = counter.total_crossings

            _draw_line_and_text(plot_img, line_tuple, count_disp)
            writer.write(plot_img)
            frame_idx += 1
    finally:
        f_csv.close()
        writer.release()
        if temp_out is not None and temp_out.is_file():
            try:
                shutil.copy2(temp_out, out_path)
            except OSError:
                pass
            try:
                temp_out.unlink()
            except OSError:
                pass
        if temp_src is not None and temp_src.is_file():
            try:
                temp_src.unlink()
            except OSError:
                pass
        run_meta["processed_frames"] = int(frame_idx)
        run_meta["line_crossings"] = int(counter.total_crossings) if counter is not None else None
        run_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump(run_meta, f_meta, ensure_ascii=False, indent=2)

    print(f"已写入: {out_path}")
    print(f"轨迹 CSV: {csv_path}")
    print(f"运行元信息: {meta_path}")
    if counter is None:
        print("未传入 --line，已跳过越线计数与虚拟线绘制。")


if __name__ == "__main__":
    main()
