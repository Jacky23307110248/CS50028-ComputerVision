# 题目一：2DGS + AIGC 多源资产生成与融合

本目录封装 **背景 garden 2DGS**、**物体 A/B/C 训练** 与 **Blender 融合导出** 脚本。  
训练在 **Task1 DSW GPU** 上运行；COLMAP 抽帧与 Blender 融合在 **本地 CPU**。

魔搭传数脚本保留在 HW3 根目录：`upload_task1_to_workspace.py`、`download_task1_on_dsw.py`。

---

## 目录结构

```
task1/
  configs/task1.yaml      # 默认步数等
  lib/                    # paths / threestudio CLI / 产物快照
  scripts/
    check_setup.py
    prepare_object_a.py   # 本地：抽帧 + COLMAP
    run_train_background.py
    run_train_object_a.py
    run_train_object_b.py
    run_train_object_c.py # Magic123 coarse + refine
    export_for_blender.py # 本地：汇总 F1 资产
  outputs/
    background_garden/
    object_a|b|c/
    fusion/
```

---

## DSW 环境（无 conda，两套 venv）

魔搭 DSW 通常只有 **系统 Python + CUDA + 预装 PyTorch**，没有 conda。  
与题目二一样，用 **`python -m venv`**，并通过 **`--system-site-packages`** 继承 DSW 自带的 torch，避免重复装 CUDA 版 PyTorch。

### 一键创建（DSW 上首次执行）

```bash
export HW3_ROOT=/mnt/workspace/HW3
cd $HW3_ROOT

python task1/scripts/setup_dsw_env.py --target all
# 仅 2DGS：  --target 2dgs
# 仅 threestudio： --target ts
# 重建 venv： --recreate
```

会创建：

| venv | 路径 | 用途 |
|------|------|------|
| 2DGS | `.venv-2dgs/` | background、object_A |
| threestudio | `.venv-ts/` | object_B DreamFusion、object_C Magic123 |

### 激活后训练

```bash
source .venv-2dgs/bin/activate
python task1/scripts/run_train_background.py
python task1/scripts/run_train_object_a.py
deactivate

source .venv-ts/bin/activate
python task1/scripts/run_train_object_b.py
python task1/scripts/run_train_object_c.py
deactivate
```

也可不 activate，直接用 venv 里的 python：

```bash
.venv-2dgs/bin/python task1/scripts/run_train_background.py
.venv-ts/bin/python task1/scripts/run_train_object_b.py
```

### 权重路径（魔搭 download 后）

| 模型 | 路径 |
|------|------|
| SD 2.1 | `cache/hf/stable-diffusion-2-1-base` |
| SD 1.5 | `cache/hf/stable-diffusion-v1-5` |
| Zero123 | `cache/hf/zero123-diffusers` |

```bash
swanlab login   # 可选；项目 CVHW3
```

---

## 数据准备

### 背景

Mip-NeRF 360 **garden**（任一路径即可）：

- `data/mipnerf360/garden/`
- 或 `data/mipnerf360/_download/360_v2/garden/`

### 物体 A（C1：相机环绕静止物体）

1. 重拍：相机绕物体转一圈，物体保持静止。
2. 放入 `data/object_A/raw/input.mp4`。
3. **本地** COLMAP：

```bash
python task1/scripts/prepare_object_a.py --no-gpu
# 或已有照片：python task1/scripts/prepare_object_a.py --images-dir path/to/photos --no-gpu
```

`repos/2d-gaussian-splatting/convert.py` 已打 COLMAP 3.13 补丁（`FeatureExtraction.use_gpu`）。

### 物体 B（纯文本）

`data/object_B/prompt.txt`（必填），可选 `negative_prompt.txt`。

### 物体 C（单图 + 文本）

- `data/object_C/rgba.png` — 去背景 RGBA
- `data/object_C/prompt.txt`（必填），可选 `negative_prompt.txt`

---

## DSW 一键检查与训练

```bash
export HW3_ROOT=/mnt/workspace/HW3
cd $HW3_ROOT

# 魔搭拉取权重与 garden（根目录脚本）
export MODELSCOPE_API_TOKEN=ms-xxxx
python download_task1_on_dsw.py --mode assets --only-missing

python task1/scripts/check_setup.py

# 首次：创建 venv（见上一节 setup_dsw_env.py）
python task1/scripts/setup_dsw_env.py --target all

# 2DGS
source .venv-2dgs/bin/activate
python task1/scripts/run_train_background.py
python task1/scripts/run_train_object_a.py   # 需先本地 COLMAP 并上传 object_a
deactivate

# threestudio
source .venv-ts/bin/activate
python task1/scripts/run_train_object_b.py
python task1/scripts/run_train_object_c.py   # coarse + refine 两阶段
deactivate
```

冒烟测试（短步数）：

```bash
python task1/scripts/run_train_background.py --smoke
python task1/scripts/run_train_object_b.py --smoke
python task1/scripts/run_train_object_c.py --smoke
```

SwanLab 项目名与题目二相同：**CVHW3**。

---

## 产物保存（供 F1 Blender + 日后 F2）

每次训练后会快照到 `task1/outputs/<任务>/artifacts/`：

| 任务 | 内容 |
|------|------|
| background / object_a | 完整 2DGS `model/`（含 point_cloud、ckpt） |
| object_b | DreamFusion trial + `last.ckpt` + parsed.yaml |
| object_c | coarse 与 refine 两套 trial |

Mesh 导出（训练脚本末尾自动 `--export`）复制到 `task1/outputs/fusion/blender/`。

---

## 本地 Blender 融合（F1）

1. 从 DSW 下载 `task1/outputs/` 整目录（或至少 `fusion/` + 各 `artifacts/`）。
2. 汇总导出：

```bash
python task1/scripts/export_for_blender.py
```

3. 阅读 `task1/outputs/fusion/BLENDER_F1.md`，在 Blender 中导入 `fusion/export/` 下 ply 与 obj，布置相机并渲染漫游视频。

第一版不做 F2 自动融合；完整 ckpt / manifest 已保留以便后续扩展。

---

## 魔搭同步（根目录）

```bash
# 本地上传 assets（garden、权重、repos）
python upload_task1_to_workspace.py --mode assets

# 单独补传 object_a / object_b / object_c 数据
python upload_task1_to_workspace.py --mode object_a
python upload_task1_to_workspace.py --mode object_b
python upload_task1_to_workspace.py --mode object_c
```

数据集：`sSzHox/task1_Data`。

---

## 建议执行顺序

1. 本地：C1 重拍 object_A → `prepare_object_a.py --no-gpu`
2. 准备 object_B/C 的 prompt 与 object_C 的 rgba.png
3. 本地上传 assets + object 数据到魔搭
4. DSW：download → check_setup → 四条训练命令
5. 下载 outputs → 本地 `export_for_blender.py` → Blender 融合与录屏
6. 实验报告（三种方法对比 + 统一表示方案 + SwanLab 曲线）
