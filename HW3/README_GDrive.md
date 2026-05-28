# 计算机视觉 HW3 · Google Drive 资源包清单

> **Google Drive 文件夹**：（请在提交前将下方链接替换为你的网盘地址）  
> `https://drive.google.com/drive/folders/YOUR_HW3_FOLDER_ID?usp=sharing`

本文件为 HW3 **Google Drive 资源包清单**（网盘根目录与 [GitHub HW3 目录](https://github.com/Jacky23307110248/CS50028-ComputerVision/tree/main/HW3) 内 `README_GDrive.md` 内容一致）。  
**源代码在 GitHub**；本网盘提供数据集、预训练权重缓存与全量训练输出。

**合并路径**（网盘路径 → 本地 `HW3/` 下同名路径）见 [`README.md`](README.md) 表格及各任务 README。若不从网盘下载，可自行获取数据，见 [`README.md` 数据准备](README.md#数据准备)。

---

## 网盘根目录应包含的顶层项

```text
HW3/                                          # 本 Drive 文件夹根
├── README_GDrive.md                           # 本文件
├── cache/                                     # HF 权重（题目一）
├── data/                                      # 全部数据集
├── task1/outputs/                             # 题目一训练与融合产物
└── task2/outputs/                             # 题目二 checkpoint 与评测
```

---

## 1. `cache/`（题目一 · 扩散模型权重）

合并到：`HW3/cache/`

```text
cache/hf/
├── stable-diffusion-2-1-base/
├── stable-diffusion-v1-5/
└── zero123-diffusers/
```

用途：threestudio（物体 B）、Magic123（物体 C）。路径与 `task1/README_task1.md` 中表格一致。

---

## 2. `data/`（题目一 + 题目二）

合并到：`HW3/data/`

### 2.1 题目一

```text
data/
├── mipnerf360/
│   └── garden/                    # 或 _download/360_v2/garden/
├── object_A/raw/                  # input.mp4 + COLMAP 产物（prepare 后）
├── object_B/prompt.txt
└── object_C/rgba.png, prompt.txt
```

### 2.2 题目二（CALVIN，LeRobot v3.0）

```text
data/calvin_task_ABC_D/
├── lerobot_0_4/                   # 环境 A
├── lerobot_1_4/                   # 环境 B
├── lerobot_2_4/                   # 环境 C
└── lerobot_3_4/                   # 环境 D（仅测试）
```

> 若网盘提供的是 v2.1 原始分卷，克隆 GitHub 后需在 Linux 执行 `bash task2/run_convert_calvin.sh` 再训练。

---

## 3. `task1/outputs/`（题目一训练输出）

合并到：`HW3/task1/outputs/`

```text
task1/outputs/
├── background_garden/
├── object_a/
├── object_b/
├── object_c/
└── fusion/                        # Blender 导出与 BLENDER_F1.md
```

各任务下 `artifacts/` 含 2DGS `model/`、threestudio trial、mesh 导出等，详见 [`task1/README_task1.md`](task1/README_task1.md#产物保存供-f1-blender--日后-f2)。

---

## 4. `task2/outputs/`（题目二训练与评测）

合并到：`HW3/task2/outputs/`

```text
task2/outputs/
├── act_B_only/                    # 实验 1：仅 B 环境训练
│   └── checkpoints/.../pretrained_model
├── act_ABC_mixed/                 # 实验 2：A+B+C 联合训练
│   └── checkpoints/.../pretrained_model
├── eval_D_B_only/                 # zero-shot 评测（示例目录名）
├── eval_D_ABC/
└── eval_compare_id_ood.csv        # 可选：汇总对比表
```

评测脚本输出：`eval_summary.json`、`eval_episodes.csv`、`eval_per_task.csv` 等，见 [`task2/README_task2.md`](task2/README_task2.md)。

---

## 5. 合并后快速验证

```bash
cd HW3
python task1/scripts/check_setup.py
python task2/scripts/check_setup.py
```

题目二完整训练/评测前需确认 CALVIN 已为 v3.0 且 `task2/configs/splits.json` 存在。
