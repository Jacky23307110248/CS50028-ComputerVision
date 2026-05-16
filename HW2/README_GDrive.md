# 计算机视觉 HW2 · Google Drive 资源包清单

**Google Drive 文件夹**：[打开 HW2 资源包](https://drive.google.com/drive/folders/1cgVH-8hv9I9M-6XvFUm0BxcQIiJ1wesb?usp=drive_link)

本文件为 HW2 **Google Drive 资源包清单**（网盘根目录与 [GitHub HW2 目录](https://github.com/Jacky23307110248/CS50028-ComputerVision/tree/main/HW2) 内 `README_GDrive.md` 内容一致）。  
**源代码在 GitHub**；本网盘提供数据集、全量训练输出与 Task 2 推理产物。

**合并路径**（网盘路径 → 本地 `HW2/` 下同名路径）见根目录 [`README.md`](README.md) 表格及各任务 README 的「从 Google Drive 合并」一节。若不从网盘下载，可自行获取数据集，见 [`README.md` 数据集获取](README.md#数据集获取)。下文仅列**网盘内应有什么**。

---

## 网盘根目录应包含的顶层项

```text
HW2/                                          # 本 Drive 文件夹根
├── README_GDrive.md                           # 本文件
├── Oxford-IIIT/                               # 数据集（Task 1 & 3）
├── evilspirit05/visdrone/versions/14/         # 数据集（Task 2）
├── task1_classification/outputs/            # Task 1 全量实验输出
├── task3_segmentation/outputs/                # Task 3 全量实验输出
└── task2_detection_tracking/
    ├── runs/
    ├── detectVideo/
    ├── weights/
    ├── yolov8n.pt
    ├── yolov8s.pt
    └── yolov8m.pt
```

---

## 1. `Oxford-IIIT/`（数据集）

**用途**：Task 1 分类、Task 3 分割。

**应包含的关键路径**（允许 `images/images/`、`annotations/annotations/` 一层嵌套）：

```text
Oxford-IIIT/
├── images/                          # 或 images/images/*.jpg
└── annotations/                     # 或 annotations/annotations/
    ├── trainval.txt
    ├── test.txt
    └── trimaps/                     # Task 3 必需；Task 1 可不使用
        └── <image_name>.png
```

---

## 2. `evilspirit05/visdrone/versions/14/`（数据集）

**用途**：Task 2 YOLOv8 微调。

**应包含的关键路径**：

```text
evilspirit05/visdrone/versions/14/
├── VisDrone2019-DET-train/VisDrone2019-DET-train/
│   ├── images/
│   ├── annotations/
│   └── labels/                      # convert_visdrone_to_yolo.py 生成
└── VisDrone2019-DET-val/VisDrone2019-DET-val/
    ├── images/
    ├── annotations/
    └── labels/
```

---

## 3. `task1_classification/outputs/`（按当前代码结构生成）

`runner.py` 每次运行都会创建一个目录：`<exp_name>_<YYYYMMDD_HHMMSS>/`。

**每个实验目录内应包含的文件**：

| 文件 / 目录 | 说明 |
|-------------|------|
| `best.pth` | 验证集最优权重 |
| `config_used.yaml` | 本次实验完整配置 |
| `metrics.json` | 汇总指标（含 `best_val_acc`、`test_acc`） |
| `curves.npz` | 逐 epoch train/val loss/accuracy |
| `figs/curves.png` | 训练曲线图 |

### 3.1 目录命名与实验族（当前版本）

当前 `task1_classification/scripts/` 提供四组批量调度脚本，对应输出目录前缀如下：

| 调度脚本 | 典型 `exp_name` 前缀 | 说明 |
|----------|----------------------|------|
| `run_ablation_suite.py` | `baseline_resnet18`、`ablation_pretrained_false`、`ablation_scratch_ep80`、`ablation_scratch_ep120` | 预训练消融 |
| `run_attention_suite.py` | `attention_se`、`attention_se_high`、`attention_cbam`、`attention_cbam_high`、`baseline_no_attention_ep40` 等 | 注意力 20/40 epoch |
| `run_hparam27.py` | `hparam27_blr*_hlr*_ep*` | 27 组网格搜索 |
| `run_transformers_suite.py` | `baseline_resnet18_aligned_transformer_adamw`、`vit_tiny_adamw`、`swin_tiny_adamw` | Transformer 对齐对比 |

其中 `hparam27` 的组合为：

| backbone_lr | head_lr | epochs |
|-------------|---------|--------|
| 0.0001 | 0.001, 0.002, 0.003 | 20, 30, 40 |
| 0.0002 | 0.001, 0.002, 0.003 | 20, 30, 40 |
| 0.0003 | 0.001, 0.002, 0.003 | 20, 30, 40 |

目录名示例：`hparam27_blr0.0003_hlr0.003_ep40_20260516_185944`（均以 `hparam27_blr` 开头）。

---

## 4. `task3_segmentation/outputs/`（全量，共 6 个 run 子目录）

**每个子目录内应包含的文件**：

| 文件 | 说明 |
|------|------|
| `best.pt` | 验证集 mIoU 最优 `state_dict` |
| `config_merged.yaml` | 运行配置快照 |
| `metrics.csv` | 列：`epoch, train_loss, train_miou, val_loss, val_miou` |
| `train.log` | 文本训练日志 |
| `result.json` | `best_val_miou`、`test_mIoU` 等 |

### 4.1 run 子目录完整列表

| 子目录名 | 损失 | epochs |
|----------|------|--------|
| `20260514_165612_unet_ce_ep30_bs8_img256_lr0.001_wd0.0001_cw1.0_dw1.0_seed42` | CE | 30 |
| `20260514_172639_unet_dice_ep30_bs8_img256_lr0.001_wd0.0001_cw1.0_dw1.0_seed42` | Dice | 30 |
| `20260514_175630_unet_ce_dice_ep30_bs8_img256_lr0.001_wd0.0001_cw1.0_dw1.0_seed42` | CE+Dice | 30 |
| `20260514_183140_unet_dice_ep50_bs8_img256_lr0.001_wd0.0001_cw1.0_dw1.0_seed42` | Dice | 50 |
| `20260514_191959_unet_ce_dice_ep50_bs8_img256_lr0.001_wd0.0001_cw1.0_dw1.0_seed42` | CE+Dice | 50 |
| `20260514_200907_unet_ce_ep50_bs8_img256_lr0.001_wd0.0001_cw1.0_dw1.0_seed42` | CE | 50 |

---

## 5. `task2_detection_tracking/`（runs + detectVideo + 权重）

### 5.1 `runs/detect/`（3 次训练实验）

每个实验目录 `<name>/` 下应包含：

| 路径 | 说明 |
|------|------|
| `args.yaml` | Ultralytics 训练参数 |
| `results.csv` | 逐 epoch 训练/验证指标（含 mAP 等） |
| `weights/best.pt` | 该次实验最优权重 |
| `weights/last.pt` | 最后一轮权重 |

**实验目录名**：

- `visdrone`
- `visdrone-2`
- `visdrone-3`

完整路径示例：`runs/detect/visdrone/weights/best.pt`。

### 5.2 `weights/`

| 文件 | 说明 |
|------|------|
| `best.pt` | 供 `infer_track.py` 默认加载的微调权重（实际上从最优模型(visdrone-3)的best.pt复制而来）|

### 5.3 `detectVideo/`

| 文件 / 目录 | 说明 |
|-------------|------|
| `input.mp4` | 测试输入视频 |
| `output.mp4` | 检测 + 跟踪（+ 越线）可视化结果 |
| `track_log.csv` | 列：`frame, track_id, cx, cy, cls, conf, x1, y1, x2, y2` |
| `track_run_meta.json` | 视频分辨率、fps、越线参数、`line_crossings` 等 |
| `frames_for_report/` | 按帧导出的 PNG（可为空目录，仅 `.gitkeep`） |

### 5.4 目录根下的 YOLO 预训练权重（可选）

| 文件 | 说明 |
|------|------|
| `yolov8n.pt` | 训练起点（nano） |
| `yolov8s.pt` | 训练起点（small） |
| `yolov8m.pt` | 训练起点（medium） |

仅 **重新训练** 时需要；仅跑 `infer_track.py` 时 `weights/best.pt` 即可。

---

## 未包含在本网盘的内容

| 内容 | 获取方式 |
|------|----------|
| Python 源代码、`configs/` | GitHub 仓库 |
| 实验报告 PDF | 课程提交系统 / 本地 `report.ipynb` 导出 |
| `report_figures/` 汇总目录 | 未上传；插图见各任务 `ReportFig/` 或上文 `outputs` / `detectVideo` |
| 虚拟环境 `.venv` | 本地 `pip install -r requirements.txt` |

---

## 合并方式

`git clone` 后与网盘内容按**同名路径**合并到 `HW2/` 根目录，无需改名。合并后完整目录树见 GitHub  [`README.md`](README.md)「仓库结构」一节。
