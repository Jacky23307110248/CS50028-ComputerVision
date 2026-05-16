# Task 3：U-Net 语义分割（Oxford-IIIT Pet）

从零实现 U-Net（无预训练），在 Oxford-IIIT Pet trimap 上做三分类分割，对比 CE / Dice / CE+Dice 三种损失。

## 从 Google Drive 合并

本任务所需大文件不在 Git 中。可选：[Google Drive 资源包](https://drive.google.com/drive/folders/1cgVH-8hv9I9M-6XvFUm0BxcQIiJ1wesb?usp=drive_link)（推荐），或与 Task 1 相同按根目录 [`README.md` 数据集获取](../README.md#数据集获取) 从 **VGG 官网** 获取 `Oxford-IIIT`（须含 `trimaps/`）。清单见 [`README_GDrive.md`](../README_GDrive.md) §1、§4。

| 网盘路径 | 本地路径 |
|----------|----------|
| `Oxford-IIIT/` | `HW2/Oxford-IIIT/` |
| `task3_segmentation/outputs/` | `HW2/task3_segmentation/outputs/` |

## 目录结构

```text
task3_segmentation/
├── runner.py           # 入口：check_data / train
├── config_schema.py    # argparse 与配置校验
├── defaults.py         # 默认超参
├── models/
│   ├── unet.py         # U-Net（编码器-解码器 + skip connection）
│   └── __init__.py
├── core/
│   ├── dataset.py      # 分割数据集与 8:2 划分
│   ├── engine.py       # 训练/验证循环
│   ├── losses.py       # CrossEntropy、Dice、组合损失
│   ├── metrics.py      # mIoU
│   └── utils.py        # 日志、CSV、run 目录
├── outputs/            # 实验输出（网盘，见 README_GDrive §4）
├── ReportFig/          # 报告用图（可选）
└── requirements.txt
```

## 环境

```bash
cd task3_segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`torch`、`torchvision`、`numpy`、`Pillow`、`PyYAML`、`matplotlib`。

## 数据集

默认路径 `../Oxford-IIIT`（`--data-root` 可改）。除 Task 1 所需文件外，须有 **`trimaps/`**（见下，网盘数据集已含）。

除 Task 1 所需的 `images/` 与 `trainval.txt` 外，分割还需要 **trimap**：

```text
Oxford-IIIT/
├── images/ ...
└── annotations/              # 或 annotations/annotations/
    ├── trainval.txt
    ├── test.txt
    └── trimaps/              # 或 trimaps 在嵌套 annotations 下
        └── <image_name>.png
```

- trimap 原始标签为 `{1,2,3}`，加载时映射为 `{0,1,2}` 三类  
- **训练/验证**：从 `trainval` 按 seed 随机 8:2 划分（`core/dataset.py`）  
- **测试**：训练结束后用官方 `test` split 评估最佳 checkpoint

## 快速开始

**检查数据与 DataLoader**

```bash
python runner.py --mode check_data
```

**训练（默认 CE，50 epoch）**

```bash
python runner.py
```

**三种损失对比（作业要求）**

```bash
# 仅 Cross-Entropy
python runner.py --loss ce --epochs 50

# 仅 Dice
python runner.py --loss dice --epochs 50

# CE + Dice
python runner.py --loss ce_dice --ce-weight 1.0 --dice-weight 1.0 --epochs 50
```

## 常用 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `train` | `train` / `check_data` |
| `--data-root` | `../Oxford-IIIT` | 数据集根目录 |
| `--output-root` | `./outputs` | 输出根目录 |
| `--run-name` | 空 | 自定义 run 文件夹名；空则按时间戳+超参自动生成 |
| `--loss` | `ce` | `ce` / `dice` / `ce_dice` |
| `--ce-weight` / `--dice-weight` | `1.0` | 组合损失权重 |
| `--img-size` | `256` | 输入边长 |
| `--batch-size` | `8` | batch size |
| `--epochs` | `50` | 训练轮数 |
| `--lr` | `1e-3` | AdamW 学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--scheduler` | `cosine` | `none` / `cosine` |
| `--patience` | `10` | 验证 mIoU 无提升则早停；`0` 关闭 |
| `--seed` | `42` | 划分与初始化随机种子 |
| `--device` | `cuda` | `cuda` 不可用时自动回退 CPU |
| `--num-workers` | `2` | DataLoader workers |
| `--save-every` | `0` | 每 N epoch 存 `epoch_N.pt`；`0` 仅保存 best |

示例：指定输出名与更长训练

```bash
python runner.py --loss ce_dice --epochs 50 --run-name unet_ce_dice_ep50 --patience 15
```

## 训练输出（`outputs/<run_name>/`）

网盘已含 6 个历史 run（见 [`README_GDrive.md`](../README_GDrive.md) §4）。本地新训练时 run 名示例：`20260514_165612_unet_ce_ep30_bs8_img256_lr0.001_wd0.0001_cw1.0_dw1.0_seed42`

| 文件 | 说明 |
|------|------|
| `config_merged.yaml` | 本次运行完整配置 |
| `train.log` | 逐 epoch 文本日志 |
| `metrics.csv` | `epoch, train_loss, train_miou, val_loss, val_miou` |
| `best.pt` | 验证集 mIoU 最优权重 |
| `result.json` | `best_val_miou`、`test_mIoU` 等汇总 |

`runner.py` 内 `save_results_plot` 已注释，不会自动生成 `results.png`；可从 `metrics.csv` 或 `train.log` 自行绘图。

## 模型与损失

- **U-Net**（`models/unet.py`）：4 层下采样 + 4 层上采样，skip connection 拼接，输出 3 类 logits，**无预训练权重**。  
- **Dice Loss**（`core/losses.py`）：对 softmax 概率与 one-hot 标签按类计算 Dice，取 `1 - mean(dice)`。  
- **评价指标**：逐 batch 平均 **mIoU**（仅对 union>0 的类求平均）。
