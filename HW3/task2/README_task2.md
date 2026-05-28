# 题目二：CALVIN + ACT

## 数据与环境划分

| 分卷 | 环境 | 用途 |
|------|------|------|
| `lerobot_0_4` | A | 训练 / 验证（仅 ABC 实验） |
| `lerobot_1_4` | B | 实验1 训练；ABC 实验亦含 |
| `lerobot_2_4` | C | 训练 / 验证（仅 ABC 实验） |
| `lerobot_3_4` | D | **仅测试**（zero-shot 离线评测） |

- A/B/C：各环境 **5% episode → val**，其余 → train（`seed=42`）。
- D：**全部 episode → test**，训练与验证阶段 **绝不加载**。
- 划分文件：`task2/configs/splits.json`（运行 `build_splits.py` 生成）。

## 实验

1. **B-only**：`python task2/scripts/train_b.py`
2. **A+B+C**：`python task2/scripts/train_abc.py`（`ConcatDataset`，不改原数据）
3. **离线评测**（D zero-shot + ID val，单次遍历多指标）：
   ```bash
   python task2/scripts/eval_zero_shot_d.py --checkpoint <pretrained_model> --split d_test --run-name eval_D_B_only
   # --split: d_test | b_val | abc_val
   bash task2/run_full_eval_suite.sh   # 4 runs after both models trained
   tail -f task2/outputs/eval_D_B_only/eval.log   # progress
   ```
   输出目录：`eval_summary.json`, `eval_episodes.csv`, `eval_per_task.csv`,
   `eval_chunk_steps.csv`, `eval_horizons.csv`, `eval.log`（已移除 batch CSV）

超参与 LeRobot `lerobot-train` 默认一致：`batch_size=8`, `steps=100000`, `log_freq=200`, `eval/save_freq=20000`。

## 数据格式（必做一次）

Hub 下载的 CALVIN 分卷为 **LeRobot v2.1**；当前 `lerobot` 训练需 **v3.0**。请在 **Linux（推荐 AMD 训练机）** 执行原地转换（四卷合计约数小时）：

```bash
bash task2/run_convert_calvin.sh
# 或：python task2/scripts/convert_calvin_to_v30.py --env all
```

> **注意**：Windows 上官方转换脚本可能在视频拼接阶段失败（`av.error.ValueError`）。本地 Windows 仅建议跑 `build_splits.py` / `check_setup.py`；完整训练与转换请在 AMD 完成。

## 本地检查

```bash
cd HW3
pip install -e "./lerobot[dataset]"
pip install -r task2/requirements_task2.txt
python task2/scripts/build_splits.py
python task2/scripts/check_setup.py
python task2/scripts/smoke_test.py   # 短跑验证（需至少 B、D 已转 v3.0）
```

## AMD（ROCm）

```bash
export HW3_ROOT=/mnt/workspace/HW3
cd $HW3_ROOT
pip install -e ./lerobot
pip install -r task2/requirements_task2.txt
swanlab login   # 可选；项目 CVHW3 / workspace 23307110248JackyH

bash task2/run_train_b.sh
bash task2/run_train_abc.sh
# 训练完成后分别评测
bash task2/run_eval_d.sh task2/outputs/act_B_only/checkpoints/.../pretrained_model
bash task2/run_eval_d.sh task2/outputs/act_ABC_mixed/checkpoints/.../pretrained_model
```

## 目录

```
task2/
  configs/splits.json
  lib/          # 划分、数据集、训练、评测
  scripts/
  outputs/      # checkpoint 与日志
```
