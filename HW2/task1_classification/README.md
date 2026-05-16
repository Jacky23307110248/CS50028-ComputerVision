# Task 1：Oxford-IIIT Pet 图像分类

在 Oxford-IIIT Pet 上微调预训练模型，覆盖以下实验族：基线、预训练消融、SE/CBAM 注意力、27 组超参数搜索、ViT-Tiny/Swin-T（timm）。

## 从 Google Drive 合并

本任务大文件不在 Git 中。可选：[Google Drive 资源包](https://drive.google.com/drive/folders/1cgVH-8hv9I9M-6XvFUm0BxcQIiJ1wesb?usp=drive_link)（推荐），或按根目录 [`README.md` 数据集获取](../README.md#数据集获取) 从 VGG 官网下载 `Oxford-IIIT`。网盘清单见 [`README_GDrive.md`](../README_GDrive.md) §1、§3。

| 网盘路径 | 本地路径 |
|----------|----------|
| `Oxford-IIIT/` | `HW2/Oxford-IIIT/` |
| `task1_classification/outputs/` | `HW2/task1_classification/outputs/` |

## 目录结构

```text
task1_classification/
├── runner.py                 # 训练 + 验证 + 官方 test 评估入口
├── test.py                   # 环境/数据管线 smoke test
├── dataset.py                # 数据加载、增强、train/val 随机划分
├── engine.py                 # 单 epoch 训练/验证
├── model_factory.py          # 模型、注意力、优化器、学习率调度
├── losses.py / metrics.py / plot_utils.py / utils.py
├── configs/                  # 实验 YAML 配置
├── scripts/                  # 批量实验调度（ablation/attention/hparam27/transformers）
├── outputs/                  # 实验输出（网盘）
├── ReportFig/                # 报告用图（可选）
└── requirements.txt
```

## 环境

```bash
cd task1_classification
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

依赖：`torch`、`torchvision`、`timm`、`PyYAML`、`matplotlib`、`Pillow`、`numpy`。

## 数据集与划分

默认数据根目录（自动探测）：

1. `../Oxford-IIIT`（推荐，与 HW2 根目录同级）
2. `task1_classification/dataset`

也可在 YAML 中设置 `data_root`。

期望目录结构（兼容一层嵌套）：

```text
Oxford-IIIT/
├── images/                   # 或 images/images/
│   └── *.jpg
└── annotations/              # 或 annotations/annotations/
    ├── trainval.txt
    └── test.txt
```

- 训练/验证：从 `trainval.txt` 按 `val_ratio`（默认 `0.1`）随机划分，随机种子由 `split_seed` 控制。  
- 测试：使用 `test.txt`，训练结束后自动加载 best checkpoint 评估并写入 `metrics.json`。

## 快速开始

```bash
# 1) 可选：先做环境/数据连通性检查（默认不下载预训练权重）
python test.py --config configs/baseline_resnet18.yaml --pretrained-mode off

# 2) 单次训练
python runner.py --config configs/baseline_resnet18.yaml
```

可选 CLI 覆盖：

```bash
python runner.py --config configs/ablation.yaml --override-exp-name ablation_pretrained_false --override-seed 42
```

## 配置文件（`configs/`）

| 文件 | 用途 |
|------|------|
| `baseline_resnet18.yaml` | ResNet-18 + ImageNet 预训练 + SGD 分层学习率 |
| `baseline_resnet18_aligned_transformer.yaml` | 与 ViT/Swin 对齐的 AdamW + cosine_warmup |
| `ablation.yaml` | 预训练消融模板 |
| `hparam_grid.yaml` | 超参候选值（由 `scripts/run_hparam27.py` 调度） |
| `attention_se.yaml` / `attention_se_high.yaml` | SE（全层 / 高层） |
| `attention_cbam.yaml` / `attention_cbam_high.yaml` | CBAM（全层 / 高层） |
| `vit_tiny.yaml` / `swin_tiny.yaml` | timm Transformer |

### 常用 YAML 字段

| 字段 | 说明 |
|------|------|
| `model_name` | `resnet18` / `resnet34` / `vit_tiny` / `swin_tiny` |
| `pretrained` | 是否加载 ImageNet 预训练 |
| `attention` | `none` / `se` / `se_high` / `cbam` / `cbam_high`（仅 ResNet） |
| `backbone_lr` / `head_lr` | 骨干与分类头学习率 |
| `optimizer` | `sgd` / `adamw` |
| `scheduler` | `cosine` / `cosine_warmup` / `step` / `none` |
| `epochs` / `batch_size` / `img_size` | 训练轮数、批大小、输入尺寸 |
| `val_ratio` / `split_seed` | `trainval` 内部划分验证集比例与种子 |

## 运行方式

### 单实验

```bash
python runner.py --config configs/baseline_resnet18.yaml
python runner.py --config configs/vit_tiny.yaml
```

### 批量实验

```bash
python scripts/run_ablation_suite.py
python scripts/run_attention_suite.py
python scripts/run_hparam27.py
python scripts/run_transformers_suite.py
```

四个脚本与产物对应关系：

| 脚本 | 作用 | 典型输出前缀 |
|------|------|--------------|
| `scripts/run_ablation_suite.py` | 基线 + 预训练消融（20/80/120） | `baseline_resnet18`、`ablation_*` |
| `scripts/run_attention_suite.py` | SE/CBAM（20 epoch + 40 epoch） | `attention_*`、`baseline_no_attention_ep40` |
| `scripts/run_hparam27.py` | 27 组学习率/epoch 网格搜索 | `hparam27_blr*_hlr*_ep*` |
| `scripts/run_transformers_suite.py` | 对齐配方下 ResNet / ViT / Swin | `baseline_resnet18_aligned_transformer_adamw`、`vit_tiny_adamw`、`swin_tiny_adamw` |

推荐执行方式（PowerShell）：

```powershell
# 复现 Task1 全量 outputs：四个批量脚本都跑一遍
python scripts/run_ablation_suite.py --seed 42 --max-parallel-jobs 4 --max-gpu-jobs 2
python scripts/run_attention_suite.py --seed 42 --max-parallel-jobs 4 --max-gpu-jobs 2
python scripts/run_hparam27.py --seed 42 --max-parallel-jobs 12 --max-gpu-jobs 8
python scripts/run_transformers_suite.py --seed 42 --max-parallel-jobs 3 --max-gpu-jobs 3
```

常用调度参数：

```bash
python scripts/run_hparam27.py --max-parallel-jobs 12 --max-gpu-jobs 8
python scripts/run_attention_suite.py --dry-run
```

### 环境变量覆盖（用于调度脚本/手工循环）

| 环境变量 | 作用 |
|----------|------|
| `T1_BACKBONE_LR` | 覆盖 `backbone_lr` |
| `T1_HEAD_LR` | 覆盖 `head_lr` |
| `T1_EPOCHS` | 覆盖 `epochs` |
| `T1_PRETRAINED` | `1` / `0` |
| `T1_ATTENTION` | 覆盖 `attention` |

`hparam27` 的候选网格：

```text
backbone_lr: 0.0001, 0.0002, 0.0003
head_lr:     0.001, 0.002, 0.003
epochs:      20, 30, 40
```

## 训练输出（`outputs/<exp_name>_<timestamp>/`）

| 文件 | 说明 |
|------|------|
| `config_used.yaml` | 本次实验完整配置 |
| `best.pth` | 验证集最优权重 |
| `metrics.json` | 汇总指标（`best_val_acc`、`test_acc` 等） |
| `curves.npz` | 逐 epoch train/val loss/accuracy |
| `figs/curves.png` | 曲线图 |

当前代码中，`runner.py` 已自动完成 test 评估并写入 `metrics.json`，无需额外 backfill 脚本。
