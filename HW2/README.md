# 计算机视觉 HW2：分类 · 检测跟踪 · 语义分割

本仓库为《计算机视觉》期中作业（HW2）**源代码**，包含 Task 1–3 的训练与推理脚本。  
**数据集、训练输出、Task 2 的 `runs/` / `detectVideo/` 等大文件不在 Git 中**，需从 Google Drive 下载后与本地仓库合并；网盘内容与目录清单见 [`README_GDrive.md`](README_GDrive.md)。

| 任务 | 代码目录 | 文档 |
|------|----------|------|
| Task 1 | [`task1_classification/`](task1_classification/) | [README](task1_classification/README.md)（含 `scripts/` 批量调度） |
| Task 2 | [`task2_detection_tracking/`](task2_detection_tracking/) | [README](task2_detection_tracking/README.md) |
| Task 3 | [`task3_segmentation/`](task3_segmentation/) | [README](task3_segmentation/README.md) |

作业说明：[`HW2_计算机视觉.md`](HW2_计算机视觉.md)

---

## 仓库结构

```text
HW2/
├── README.md                 # 本文件（GitHub 入口）
├── README_GDrive.md          # 网盘资源清单（GitHub 与 Google Drive 各一份，内容一致）
├── HW2_计算机视觉.md
├── report.ipynb
├── Oxford-IIIT/              # 网盘：Task 1 & 3 数据集
├── evilspirit05/             # 网盘：Task 2 VisDrone（见 visdrone/versions/14/）
├── task1_classification/     # Git：代码；网盘：outputs/
├── task2_detection_tracking/ # Git：代码；网盘：runs/、detectVideo/、weights/、yolov8*.pt
└── task3_segmentation/       # Git：代码；网盘：outputs/
```

克隆 GitHub 后，将 [Google Drive 资源包](https://drive.google.com/drive/folders/1cgVH-8hv9I9M-6XvFUm0BxcQIiJ1wesb?usp=drive_link) 中对应文件夹**按同名路径合并**到上表位置。各任务本地放置路径见下表及子 README；`outputs` / `runs` 子目录的完整列表见 [`README_GDrive.md`](README_GDrive.md)。

| 网盘路径 | 合并到 `HW2/` 下 | 任务 | 放置说明 |
|----------|------------------|------|----------|
| `Oxford-IIIT/` | `Oxford-IIIT/` | 1、3 | [task1](task1_classification/README.md#从-google-drive-合并) · [task3](task3_segmentation/README.md#从-google-drive-合并) |
| `evilspirit05/visdrone/versions/14/` | 同左 | 2 | [task2](task2_detection_tracking/README.md#从-google-drive-合并) |
| `task1_classification/outputs/` | `task1_classification/outputs/` | 1 | [task1](task1_classification/README.md#从-google-drive-合并) |
| `task3_segmentation/outputs/` | `task3_segmentation/outputs/` | 3 | [task3](task3_segmentation/README.md#从-google-drive-合并) |
| `task2_detection_tracking/runs/` | 同左 | 2 | [task2](task2_detection_tracking/README.md#从-google-drive-合并) |
| `task2_detection_tracking/detectVideo/` | 同左 | 2 | 同上 |
| `task2_detection_tracking/weights/` | 同左 | 2 | 同上 |
| `task2_detection_tracking/yolov8*.pt` | `task2_detection_tracking/` 根下 | 2（可选） | 同上 |

---

## 数据集获取

除 [Google Drive 资源包](https://drive.google.com/drive/folders/1cgVH-8hv9I9M-6XvFUm0BxcQIiJ1wesb?usp=drive_link) 外，也可自行下载原始数据集并按下方说明放置。目录细节见 [`README_GDrive.md`](README_GDrive.md) §1、§2；合并后本地路径见上表。

### 方式 A：Google Drive

一次下载已整理好的 `Oxford-IIIT/`、`evilspirit05/visdrone/versions/14/`（含 YOLO `labels/`）及训练输出，与仓库路径一致，无需再转换。

### 方式 B：Oxford-IIIT Pet（Task 1 & 3）— 官网下载

- **下载入口**：[Oxford-IIIT Pet Dataset（VGG）](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- 在该页面选择下载方式（合计约 800 MB）：
  - **HTTP**：`images.tar.gz`（图像）与 `annotations.tar.gz`（标注，含 trimap）
  - **BitTorrent**：页面提供的 Academic Torrents 种子（官方推荐，通常更快）

**放置步骤**

1. 在 `HW2/` 下新建目录 `Oxford-IIIT/`。
2. 将 `images.tar.gz` 解压到 `Oxford-IIIT/images/`（若多出一层 `images/images/`，本仓库代码可自动识别）。
3. 将 `annotations.tar.gz` 解压到 `Oxford-IIIT/annotations/`（若出现 `annotations/annotations/`，代码同样兼容）。
4. 确认存在 `trainval.txt`、`test.txt`；Task 3 还需 `trimaps/*.png`（在 annotations 包内）。

### 方式 C：VisDrone 2019 DET（Task 2）— Kaggle

本仓库磁盘上的数据按 **VisDrone2019-DET** 官方目录名整理；与 Kaggle 上常见的 **`visdrone2019-det`** 数据集对应（勿与仅含扁平 `VisDrone/images/train` 的其他 VisDrone 镜像包混淆）。

- 示例页面（Kaggle 上存在多个上传者的同名/近名镜像，任选其一即可）：
  - [visdrone2019-det（shisuiotsutsuki）](https://www.kaggle.com/datasets/shisuiotsutsuki/visdrone2019-det)
  - 或在 Kaggle 搜索 `visdrone2019-det` / `VisDrone2019 DET`
- 需安装 [Kaggle CLI](https://github.com/Kaggle/kaggle-api) 并配置 API Token，在网页端 Download 亦可。

**放置与转换步骤**

1. 下载并解压，使 train/val 的 **官方结构** 位于：

   ```text
   HW2/evilspirit05/visdrone/versions/14/
   ├── VisDrone2019-DET-train/VisDrone2019-DET-train/
   │   ├── images/
   │   └── annotations/
   └── VisDrone2019-DET-val/VisDrone2019-DET-val/
       ├── images/
       └── annotations/
   ```

   若 Kaggle 包内目录名不同，请手动调整成上述结构（与 `convert_visdrone_to_yolo.py` 一致）。

2. 在 `images` 同级生成 YOLO 格式 `labels/`：

   ```bash
   cd task2_detection_tracking
   python scripts/convert_visdrone_to_yolo.py
   ```

3. 编辑 `configs/visdrone_local.yaml`，将 `path` 改为本机路径，例如（相对 `configs/`）：

   ```yaml
   path: ../../evilspirit05/visdrone/versions/14
   ```

也可从 [VisDrone 官方仓库](https://github.com/visdrone/visdrone-dataset) / [竞赛主页](http://aiskyeye.com/) 获取原始 zip，解压后同样执行第 2、3 步。

> `evilspirit05/visdrone/versions/14/` 下的 `data.yaml`、`new_data.yaml` 为旧 Notebook 残留（Kaggle 路径 `/kaggle/input/...`），**训练不使用**；以 `visdrone_local.yaml` 为准。

---

## 环境要求

- **Python**：建议 3.10 或 3.11  
- **GPU**：Task 1 / 3 建议 NVIDIA GPU；Task 2 可用 CPU 但较慢  

各任务目录下 `pip install -r requirements.txt`，详见子 README。

---

## 复现当前 outputs/runs 全量结果（需先完成网盘合并）

### Task 1（分类）

Task 1 的四个批量脚本用法见 [`task1_classification/README.md`](task1_classification/README.md)「批量实验」一节。  
复现当前 `task1_classification/outputs/` 的做法是四个脚本都跑一遍：

```powershell
cd task1_classification
python scripts/run_ablation_suite.py --seed 42 --max-parallel-jobs 4 --max-gpu-jobs 2
python scripts/run_attention_suite.py --seed 42 --max-parallel-jobs 4 --max-gpu-jobs 2
python scripts/run_hparam27.py --seed 42 --max-parallel-jobs 12 --max-gpu-jobs 8
python scripts/run_transformers_suite.py --seed 42 --max-parallel-jobs 3 --max-gpu-jobs 3
```

### Task 2（检测+跟踪+越线）

1) 按顺序微调 YOLOv8m / YOLOv8s / YOLOv8n（50 epoch, batch=8, imgsz=640）：

```powershell
cd task2_detection_tracking
python scripts/train.py --model yolov8m.pt --epochs 50 --batch 8 --imgsz 640 --name visdrone-3
python scripts/train.py --model yolov8s.pt --epochs 50 --batch 8 --imgsz 640 --name visdrone-2
python scripts/train.py --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --name visdrone
```

2) 从三次训练中选最佳（当前结果对应 `visdrone-3`），复制权重：

```powershell
copy runs\detect\visdrone-3\weights\best.pt weights\best.pt
```

3) 运行视频跟踪与越线计数（与当前 `detectVideo` 一致的参数）：

```powershell
python scripts/infer_track.py --weights weights/best.pt --imgsz 640 --tracker bytetrack.yaml --line 120 100 560 100
```

> 上述线段坐标对应 `640x360` 视频；输出 `detectVideo/output.mp4`、`track_log.csv`、`track_run_meta.json`。

### Task 3（分割）

按要求使用 `--scheduler cosine`，并对三种损失在 30/50 epoch 各跑一遍（共 6 次）：

```powershell
cd task3_segmentation

python runner.py --scheduler cosine --loss ce      --epochs 30
python runner.py --scheduler cosine --loss dice    --epochs 30
python runner.py --scheduler cosine --loss ce_dice --ce-weight 1.0 --dice-weight 1.0 --epochs 30

python runner.py --scheduler cosine --loss ce      --epochs 50
python runner.py --scheduler cosine --loss dice    --epochs 50
python runner.py --scheduler cosine --loss ce_dice --ce-weight 1.0 --dice-weight 1.0 --epochs 50
```

可选先做数据检查：

```powershell
python runner.py --mode check_data
```

---

## 实验报告与链接

- 报告提交：**PDF**（可由 [`report.ipynb`](report.ipynb) 导出）  
- 报告需包含：[GitHub HW2 代码](https://github.com/Jacky23307110248/CS50028-ComputerVision/tree/main/HW2)、[Google Drive 资源包](https://drive.google.com/drive/folders/1cgVH-8hv9I9M-6XvFUm0BxcQIiJ1wesb?usp=drive_link)、实验设置与结果图  

### GitHub 仓库

- **HW2 目录**：[https://github.com/Jacky23307110248/CS50028-ComputerVision/tree/main/HW2](https://github.com/Jacky23307110248/CS50028-ComputerVision/tree/main/HW2)
- **完整仓库**：<https://github.com/Jacky23307110248/CS50028-ComputerVision>

```bash
git clone https://github.com/Jacky23307110248/CS50028-ComputerVision.git
cd CS50028-ComputerVision/HW2
```
