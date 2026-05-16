# Task 2：VisDrone 检测 + 视频多目标跟踪 + 越线计数

在 VisDrone 上微调 YOLOv8，对自录视频做检测与 ByteTrack 跟踪，可选虚拟越线计数，并支持导出连续帧用于报告。

## 从 Google Drive 合并

本任务所需大文件不在 Git 中。可选： [Google Drive 资源包](https://drive.google.com/drive/folders/1cgVH-8hv9I9M-6XvFUm0BxcQIiJ1wesb?usp=drive_link)（推荐），或按根目录 [`README.md` 数据集获取](../README.md#数据集获取) 从 **Kaggle `visdrone2019-det`**（或 VisDrone 官方源）下载并运行 `convert_visdrone_to_yolo.py`。清单见 [`README_GDrive.md`](../README_GDrive.md) §2、§5。

| 网盘路径 | 本地路径 |
|----------|----------|
| `evilspirit05/visdrone/versions/14/` | `HW2/evilspirit05/visdrone/versions/14/` |
| `task2_detection_tracking/runs/` | `HW2/task2_detection_tracking/runs/` |
| `task2_detection_tracking/detectVideo/` | `HW2/task2_detection_tracking/detectVideo/` |
| `task2_detection_tracking/weights/` | `HW2/task2_detection_tracking/weights/` |
| `task2_detection_tracking/yolov8*.pt` | `HW2/task2_detection_tracking/` 根下（可选，重训用） |

`infer_track.py` 默认加载 `weights/best.pt`；若网盘无 `weights/` 可从 `runs/detect/visdrone/weights/best.pt` 复制。

## 目录结构

```text
task2_detection_tracking/
├── scripts/
│   ├── convert_visdrone_to_yolo.py   # VisDrone txt → YOLO labels
│   ├── train.py                      # YOLOv8 微调
│   ├── infer_track.py                # 视频检测 + 跟踪 + 越线
│   └── export_frames.py              # 按帧号导出 PNG
├── src/
│   ├── paths.py                      # 数据/视频/权重路径约定
│   └── line_counter.py               # 越线计数逻辑
├── configs/
│   └── visdrone_local.yaml           # Ultralytics 数据集配置
├── detectVideo/                      # 网盘（README_GDrive §5.3）
├── weights/                          # 网盘（README_GDrive §5.2）
├── runs/detect/                      # 网盘（README_GDrive §5.1）
├── yolov8n.pt / yolov8s.pt / ...     # 可选：网盘或 Ultralytics 下载
├── ReportFig/                        # 报告用图（可选）
└── requirements.txt
```

数据默认根目录：`../evilspirit05/visdrone/versions/14`（`src/paths.py` → `EVILSPIRIT14`）。合并网盘后请修改 `configs/visdrone_local.yaml` 的 `path` 指向本机路径。

## 环境

```bash
cd task2_detection_tracking
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`ultralytics`、`opencv-python`、`numpy`、`Pillow`。

## 1. 数据准备（已有网盘数据可跳过转换）

若已从网盘获得带 `labels/` 的 VisDrone 目录，可跳过本节，仅需改好 `visdrone_local.yaml` 的 `path`。

否则，在 VisDrone `images` 同级生成 YOLO 格式 `labels/`（不复制图片）：

```bash
python scripts/convert_visdrone_to_yolo.py
```

默认 `--root` 为 `HW2/evilspirit05/visdrone/versions/14`。数据在其他位置时：

```bash
python scripts/convert_visdrone_to_yolo.py --root D:\path\to\visdrone\versions\14
```

转换规则：跳过 `score==0` 的框；类别保留 VisDrone 的 1–10，写入 YOLO 时为 0–9。

### 修改训练用 data yaml

编辑 `configs/visdrone_local.yaml`，将 `path` 设为**数据集根目录**（含 `VisDrone2019-DET-train/...` 的那一层），例如：

```yaml
# 相对 configs/ 目录
path: ../../evilspirit05/visdrone/versions/14

train: VisDrone2019-DET-train/VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/VisDrone2019-DET-val/images
nc: 10
```

仓库内示例可能仍为 Linux 绝对路径（`/mnt/workspace/...`），在 Windows 本地训练前**必须**改成本机路径。

## 2. 训练

为复现当前 `runs/detect/`，按以下顺序训练三组模型（同参数，不同 backbone）：

```bash
python scripts/train.py --model yolov8m.pt --epochs 50 --batch 8 --imgsz 640 --name visdrone-3
python scripts/train.py --model yolov8s.pt --epochs 50 --batch 8 --imgsz 640 --name visdrone-2
python scripts/train.py --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --name visdrone
```

常用参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | `configs/visdrone_local.yaml` | 数据集配置 |
| `--model` | `yolov8n.pt` | 预训练权重 |
| `--epochs` | `50` | 训练轮数 |
| `--batch` | `8` | batch size |
| `--imgsz` | `640` | 输入尺寸 |
| `--name` | `visdrone` | `runs/detect/<name>/` 子目录名 |
| `--device` | 自动 | 如 `0` 或 `cpu` |

训练结束后，从三次实验中选择最佳权重（当前结果对应 `visdrone-3`）并复制到推理默认路径：

```bash
copy runs\detect\visdrone-3\weights\best.pt weights\best.pt
```

（若你本地最优实验目录不同，请对应修改路径。）

## 3. 视频检测 + 跟踪

1. 将 10–30 秒测试视频放到 `detectVideo/input.mp4`（或通过 `--source` 指定）。
2. 仅检测与跟踪（不画线、不计数）：

```bash
python scripts/infer_track.py --weights weights/best.pt
```

3. 带虚拟越线（像素坐标 `x1 y1 x2 y2`，须与视频分辨率匹配，线段宜横穿目标运动区域）：

```bash
python scripts/infer_track.py --weights weights/best.pt --imgsz 640 --tracker bytetrack.yaml --line 120 100 560 100
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--source` | `detectVideo/input.mp4` | 输入视频 |
| `--output` | `detectVideo/output.mp4` | 输出视频 |
| `--csv` | `detectVideo/track_log.csv` | 轨迹 CSV |
| `--tracker` | `bytetrack.yaml` | Ultralytics 跟踪器配置 |
| `--imgsz` | `640` | 推理尺寸 |
| `--line` | 无 | 四个浮点数，不传则跳过越线 |

**输出说明**

- `output.mp4`：检测框、类别、Tracking ID；若传入 `--line` 则叠加黄线与 `line_crossings=N`
- `track_log.csv` 列：`frame, track_id, cx, cy, cls, conf, x1, y1, x2, y2`
- `track_run_meta.json`：分辨率、fps、越线参数、最终计数等

**越线规则**（`src/line_counter.py`）：以检测框中心点相对有向直线的侧别变化判断穿越；**每个 `track_id` 仅计第一次有效穿线**。

## 4. 导出报告用连续帧

```bash
python scripts/export_frames.py --frames 120 121 122 123
python scripts/export_frames.py --video detectVideo/input.mp4 --outdir detectVideo/frames_for_report
```

帧号为 **0-based**；PNG 保存为 `frame_000120.png` 等。

## Windows 注意事项

- 路径含中文时，`infer_track.py` 会将视频复制到临时 ASCII 路径再读，避免 OpenCV 打不开文件。
- `--line` 坐标必须在画面 `宽×高` 内或与画面有交集，否则黄线不显示且计数可能恒为 0。
- `.gitignore` 默认忽略 `runs/`、`weights/*.pt`、部分 `detectVideo` 大文件；提交代码前请单独上传权重到网盘。

## 类别（nc=10）

pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor（与 `visdrone_local.yaml` 中 `names` 一致）
