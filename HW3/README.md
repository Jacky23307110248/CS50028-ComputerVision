# 计算机视觉 HW3：2DGS + AIGC 融合 · LeRobot ACT 跨环境泛化

本目录为《计算机视觉》**期末作业（HW3）**源代码，包含题目一（2D Gaussian Splatting + AIGC 多源 3D 资产与场景融合）与题目二（基于 LeRobot 的 ACT 策略跨环境泛化）。

**GitHub 仓库仅包含代码与配置**；下列大体积内容**不在 Git 中**，需从 Google Drive（或自行下载）合并到本地同名路径：

| 路径 | 说明 |
|------|------|
| `data/` | Mip-NeRF 360 garden、object A/B/C 输入、CALVIN 四环境分卷等 |
| `cache/` | Hugging Face 扩散模型权重（SD 1.5/2.1、Zero123 等） |
| `task1/outputs/` | 2DGS / threestudio 训练产物与 Blender 融合导出 |
| `task2/outputs/` | ACT checkpoint、评测结果与 SwanLab 日志 |

网盘目录清单与合并说明见 [`README_GDrive.md`](README_GDrive.md)。

作业说明：[`HW3_计算机视觉.md`](HW3_计算机视觉.md) · PDF：[`HW3_计算机视觉.pdf`](HW3_计算机视觉.pdf)

---

## 任务索引

| 题目 | 代码目录 | 详细文档 |
|------|----------|----------|
| 题目一：2DGS + AIGC 融合 | [`task1/`](task1/) | [`task1/README_task1.md`](task1/README_task1.md) |
| 题目二：CALVIN + ACT | [`task2/`](task2/) | [`task2/README_task2.md`](task2/README_task2.md) |

---

## 仓库结构

```text
HW3/
├── README.md                      # 本文件（GitHub 入口）
├── README_GDrive.md               # 网盘资源清单
├── HW3_计算机视觉.md / .pdf       # 作业说明
├── cache/                         # 网盘：HF 权重（不入 Git）
├── data/                          # 网盘：数据集（不入 Git）
├── repos/                         # Git：2d-gaussian-splatting、threestudio（源码快照）
├── lerobot/                       # Git：LeRobot 框架（题目二，已去除嵌套 .git）
├── scripts/                       # Git：通用脚本（若有）
├── task1/                         # Git：代码；网盘：outputs/
│   ├── configs/
│   ├── lib/
│   ├── scripts/
│   └── outputs/                   # 不入 Git
├── task2/                         # Git：代码；网盘：outputs/
│   ├── configs/
│   ├── lib/
│   ├── scripts/
│   └── outputs/                   # 不入 Git
├── upload_task1_to_workspace.py   # 魔搭 ModelScope 同步（可选）
├── download_task1_on_dsw.py
├── upload_calvin_to_workspace.py
└── download_calvin_on_dsw.py
```

克隆 [课程仓库](https://github.com/Jacky23307110248/CS50028-ComputerVision) 后，进入 `HW3/`，将网盘中对应文件夹**按同名路径合并**到上表位置。

---

## 环境要求

- **Python**：3.10+（推荐 3.10 / 3.11）
- **题目一 GPU 训练**：CUDA（魔搭 DSW 或本地 NVIDIA）；本地 COLMAP / Blender 可在 CPU 完成
- **题目二训练**：Linux + ROCm（本仓库在 AMD 训练机验证）或 CUDA；Windows 仅建议跑划分与检查脚本
- **实验日志**：SwanLab（可选，项目名 **CVHW3**）

### 题目二：LeRobot + ACT（本地检查）

在 **`HW3/` 根目录**执行：

```bash
cd HW3
pip install -e "./lerobot[dataset]"
pip install -r task2/requirements_task2.txt
python task2/scripts/build_splits.py
python task2/scripts/check_setup.py
```

冒烟（需 `data/calvin_task_ABC_D/` 已转 LeRobot v3.0 且含 B、D 环境）：

```bash
python task2/scripts/smoke_test.py
```

依赖说明见 [`task2/requirements_task2.txt`](task2/requirements_task2.txt)；LeRobot 本体见 [`lerobot/README.md`](lerobot/README.md)。

### 题目一：两套 venv（DSW / Linux GPU）

题目一**不要**与题目二共用同一 venv。在 GPU 机器上首次创建环境：

```bash
export HW3_ROOT=/path/to/HW3
cd $HW3_ROOT
python task1/scripts/setup_dsw_env.py --target all
```

| venv | 路径 | 用途 |
|------|------|------|
| 2DGS | `.venv-2dgs/` | 背景 garden、物体 A |
| threestudio | `.venv-ts/` | 物体 B（DreamFusion）、物体 C（Magic123） |

依赖分文件：[`task1/requirements_2dgs.txt`](task1/requirements_2dgs.txt)、[`task1/requirements_threestudio.txt`](task1/requirements_threestudio.txt)、[`task1/requirements_task1.txt`](task1/requirements_task1.txt)。

---

## 数据准备

### 从 Google Drive 合并（推荐）

见 [`README_GDrive.md`](README_GDrive.md)。合并后应至少具备：

- `data/mipnerf360/garden/`（或 `_download/360_v2/garden/`）
- `data/object_A/`、`data/object_B/`、`data/object_C/`（题目一输入）
- `data/calvin_task_ABC_D/lerobot_{0,1,2,3}_4/`（题目二，**v3.0** 格式）
- `cache/hf/stable-diffusion-2-1-base/` 等（题目一 AIGC）

### 题目二：CALVIN 自行获取

1. 从 Hugging Face Hub 下载 CALVIN 对应分卷（环境 A/B/C/D → `lerobot_0_4` … `lerobot_3_4`），放入 `data/calvin_task_ABC_D/`。
2. 在 **Linux** 上转换为 LeRobot v3.0（Windows 可能在视频拼接阶段失败）：

```bash
bash task2/run_convert_calvin.sh
# 或：python task2/scripts/convert_calvin_to_v30.py --env all
```

3. 生成 train/val/test 划分：

```bash
python task2/scripts/build_splits.py
```

输出：`task2/configs/splits.json`（已提交则无需重跑，除非改 `seed`）。

### 题目一：数据自行准备

详见 [`task1/README_task1.md`](task1/README_task1.md)：

- **背景**：Mip-NeRF 360 `garden`
- **物体 A**：`data/object_A/raw/input.mp4` → 本地 `prepare_object_a.py`（COLMAP）
- **物体 B**：`data/object_B/prompt.txt`
- **物体 C**：`data/object_C/rgba.png` + `prompt.txt`

---

## 训练与测试命令（可直接复制）

**运行目录**：以下命令均默认在 **`HW3/`** 根目录执行。

### 题目二

| 步骤 | 命令 |
|------|------|
| 实验 1：仅环境 B 训练 | `python task2/scripts/train_b.py` 或 `bash task2/run_train_b.sh` |
| 实验 2：A+B+C 联合训练 | `python task2/scripts/train_abc.py` 或 `bash task2/run_train_abc.sh` |
| D 环境 zero-shot 评测 | `python task2/scripts/eval_zero_shot_d.py --checkpoint <pretrained_model> --split d_test --run-name eval_D_B_only` |
| 全套评测（两模型训练完成后） | `bash task2/run_full_eval_suite.sh` |

`--split` 可选：`d_test` | `b_val` | `abc_val`。  
Checkpoint 默认目录：`task2/outputs/act_B_only/`、`task2/outputs/act_ABC_mixed/`。

超参（与 LeRobot 默认一致）：`batch_size=8`，`steps=100000`，`log_freq=200`，`eval/save_freq=20000`。详见 [`task2/README_task2.md`](task2/README_task2.md)。

### 题目一

| 步骤 | 命令 |
|------|------|
| 环境检查 | `python task1/scripts/check_setup.py` |
| 背景 2DGS | `python task1/scripts/run_train_background.py` |
| 物体 A 2DGS | `python task1/scripts/run_train_object_a.py` |
| 物体 B 文本生成 | `python task1/scripts/run_train_object_b.py` |
| 物体 C 单图生成 | `python task1/scripts/run_train_object_c.py` |
| 冒烟（短步数） | 各训练脚本加 `--smoke` |
| Blender 融合导出（本地） | `python task1/scripts/export_for_blender.py` |

DSW 上需先 `source .venv-2dgs/bin/activate` 或 `.venv-ts/bin/activate` 再运行对应脚本。完整流程见 [`task1/README_task1.md`](task1/README_task1.md)。

---

## 模型权重（网盘）

训练得到的最优权重与完整 `outputs/` 请上传至云盘，并在实验报告中注明链接。建议打包路径：

- `task2/outputs/act_B_only/checkpoints/.../pretrained_model`
- `task2/outputs/act_ABC_mixed/checkpoints/.../pretrained_model`
- `task1/outputs/`（background、object_a/b/c、fusion）

清单见 [`README_GDrive.md`](README_GDrive.md)。

---

## 魔搭 / DSW 同步（可选）

根目录脚本用于 ModelScope 数据集 `sSzHox/task1_Data` 与 CALVIN 工作区同步，**非作业必跑项**：

```bash
python download_task1_on_dsw.py --mode assets --only-missing
python upload_task1_to_workspace.py --mode assets
python download_calvin_on_dsw.py   # 见脚本 --help
```

---

## 与课程仓库的关系

本目录位于课程总仓库 [`CS50028-ComputerVision`](https://github.com/Jacky23307110248/CS50028-ComputerVision) 的 **`HW3/`** 子路径，与 [`HW1/`](../HW1/README.md)、[`HW2/`](../HW2/README.md) 并列。

`.gitignore` 已排除 `data/`、`cache/`、`task1/outputs/`、`task2/outputs/`；复制到课程仓库时请使用 README 末尾的 `robocopy` 命令，避免误提交约 60GB+ 大文件。
