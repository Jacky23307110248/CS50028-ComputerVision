# HW1: 从零实现三层 MLP 分类 Fashion-MNIST

本项目使用 `NumPy` 从零实现三层神经网络（MLP）分类器，包含手写自动微分与反向传播，并在 Fashion-MNIST 上完成训练、超参数搜索、测试评估与可视化分析。

**运行目录**：下文命令均默认在课程仓库的 **`HW1/` 子目录**下执行（与 `train.py`、`test.py` 同级）。若 clone 后位于仓库根目录，请先执行 `cd HW1` 再运行脚本。

---

## 模型权重说明

**本作业最终使用的训练权重已放在仓库内的 `TuningRound6/artifacts/`，不依赖 Google Drive 等第三方网盘，可直接使用下列路径或 GitHub 直链。**

- **GitHub 网页查看（以 `main` 分支为例）**：`https://github.com/Jacky23307110248/CS50028-ComputerVision/blob/main/TuningRound6/artifacts/best_model_full.npy`
- **Raw 直链（下载）**：`https://github.com/Jacky23307110248/CS50028-ComputerVision/raw/main/TuningRound6/artifacts/best_model_full.npy`

若 fork 后默认分支不是 `main`，请将 URL 中的 `main` 换成实际分支名；若作业路径不在仓库根下的 `TuningRound6/`，请把路径段改成与目录一致。


使用已提交权重复现测试：

```bash
python test.py --data_path ./FashionMNIST/raw --model_path ./TuningRound6/artifacts/best_model_full.npy --history_path ./TuningRound6/artifacts/history_full.csv --fig_dir ./figures
```

---

## 1. 环境依赖

- Python 3.9+（推荐 3.10/3.11）
- numpy
- matplotlib
- seaborn

安装依赖：

```bash
pip install numpy matplotlib seaborn
```

## 2. 项目结构（本 `HW1` 目录）

- `dataloader.py`：读取 `idx.gz` 数据并做归一化、one-hot 编码
- `model.py`：模型定义、手写 Tape 自动微分、反向传播、损失函数
- `train.py`：训练主流程（SGD + LR Decay + L2 + Early Stopping）
- `grid_search.py`：网格搜索超参数并用最佳参数进行最终训练
- `test.py`：加载最优权重，输出测试精度、混淆矩阵、错例与权重可视化
- `grad_check.py`：数值梯度检查，验证反向传播正确性
- `utils.py`：初始化、指标、绘图、模型/日志保存等通用工具
- `FashionMNIST/raw/`：Fashion-MNIST 原始 `idx.gz` 数据
- `TuningRound1/` ~ `TuningRound6/`：每轮调参独立结果目录（含 `artifacts/` 与 `figures/`）
- `reportTeX/`：实验报告 LaTeX 源文件与编译产物（`HW1.tex` / `HW1.pdf`）
- `Report.pdf`：实验报告文件，与 `reportTeX/HW1.pdf` 内容一致，置于本目录根下便于浏览
- `ExtraData/`：额外导出的中间数据（如 `grad_check_result.json`）

## 3. 数据准备

本仓库默认使用本地数据目录：`./FashionMNIST/raw`。
请确认该目录下存在以下 4 个文件：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

若 clone 后该目录为空，请从 [Fashion-MNIST 官方仓库](https://github.com/zalandoresearch/fashion-mnist) 获取上述四个 `*.gz`，放入 `FashionMNIST/raw/`。

若数据放在其他位置，请在命令中通过 `--data_path` 指定路径。

## 4. 训练与搜索

### 4.1 直接训练（默认启用 训练/验证划分并按验证集选优）

```bash
python train.py --data_path ./FashionMNIST/raw --lr 0.1 --hidden_dim 256 --batch_size 128 --l2 1e-4 --epochs 50 --patience 7 --activation relu --seed 42 --split_seed 42 --model_path ./artifacts/best_model_val.npy --history_path ./artifacts/history_val.csv --best_meta_path ./artifacts/best_meta_val.json --config_path ./artifacts/run_config_val.json --split_path ./artifacts/split_indices.npz
```

参数说明：
- `--seed`：控制参数初始化与 mini-batch 打乱
- `--split_seed`：控制 train/val 划分；设定后可与 `--seed` 解耦，减少划分噪声与初始化噪声混叠
- 若不传 `--split_seed`，训练脚本默认回退到 `seed`，保持向后兼容

如需关闭验证集划分：

```bash
python train.py --data_path ./FashionMNIST/raw --no_validation_split
```

### 4.2 网格搜索 + 最终训练（默认自动进行 full-data 重训）

```bash
python grid_search.py --data_path ./FashionMNIST/raw
```

说明：
- 搜索阶段使用验证集（默认 `val_ratio=0.2`）找到最佳超参数
- 搜索阶段默认固定 `--split_seed=42`，使不同 seed 的比较主要反映初始化/优化随机性，而不是数据划分变化
- 默认会自动执行 `train+val` 全量数据重训，产出最终提交模型 `best_model_full.npy`
- 若仅想做快速搜索（不重训），可加 `--no_retrain_full_data`
- 会额外输出 `search_summary.json`，记录最佳配置与 `top-k` 候选配置
- `best_model_val.npy` 的“best”含义是**验证阶段最优模型**：验证集准确率优先，`val_total_loss` 作为并列 tie-break，用于调参与阶段性对比
- `best_model_full.npy` 的含义是**最终提交模型**：固定最佳超参数后在 `train+val` 上重训得到，无验证集参与选优
- 最终测试与报告指标仅使用 `best_model_full.npy`；`best_model_val.npy` 不用于最终测试结果声明
- 最终训练 seed 不再固定使用 `args.seed`，而是自动使用最佳超参数组合下，单 seed 验证精度最高的那个 seed
- 当前仓库已将 6 轮实验结果按目录保存为 `TuningRound1` ~ `TuningRound6`

自定义搜索空间示例：

```bash
python grid_search.py --data_path ./FashionMNIST/raw --split_seed 42 --lr_list 0.005,0.01,0.05 --hidden_list 128,256 --batch_list 64,128 --l2_list 1e-4,1e-3 --act_list relu,sigmoid
```

## 5. 测试与可视化

```bash
python test.py --data_path ./FashionMNIST/raw --model_path ./artifacts/best_model_full.npy --history_path ./artifacts/history_full.csv --fig_dir ./figures --error_cases 5
```

复现某一轮（例如第 6 轮），建议在该目录执行：

```bash
cd TuningRound6
python ../test.py --data_path ../FashionMNIST/raw --fig_dir ./figures
```

默认行为说明：
- 默认优先加载 `best_model_full.npy`
- 若 full 模型不存在，会自动回退到 `best_model_val.npy`

该脚本会输出：
- 测试集 Accuracy（终端打印）
- 混淆矩阵（计数 + 归一化）
- 各类别 Precision / Recall / F1（终端打印）
- 第一层权重可视化图
- 错例图片与错例 CSV
- 训练历史曲线 `figures/history.png`：`plot_history` 会读取磁盘上存在的日志 CSV。默认可通过 `--history_path`（常为 `history_full.csv`）与 `--fallback_history_path`（常为 `history_val.csv`）指定；若 `fallback_history_path` 对应文件存在，将优先用它绘图，便于画出含验证集、与选模阶段一致的曲线。

## 6. 梯度检查

```bash
python grad_check.py --data_path ./FashionMNIST/raw --activation relu --dtype float64 --eps 1e-5 --tol 1e-6 --checks_per_param 8 --batch_size 16 --hidden_dim 64 --seed 42 --save_json_path ./ExtraData/grad_check_result.json
```

若输出 `PASS`，说明当前配置下数值梯度与解析梯度一致性良好。

该脚本现在会在终端打印逐点结果，并额外保存 `JSON`（默认 `./ExtraData/grad_check_result.json`），其中包含：
- 运行设置（数据路径、激活函数、`eps`、`tol`、`seed` 等）
- `W1 / b1 / W2 / b2` 的随机抽样索引与每个点的 `g_num / g_ana / rel_err`
- 最终汇总（`max_rel_err`、`mean_rel_err`、`PASS/FAIL`）

```bash
python grad_check.py --save_json_path ""
```

## 7. 主要输出文件说明（按当前目录组织）

每一轮调参目录（`TuningRoundX`）下都包含：

- `TuningRoundX/artifacts/best_model_val.npy`：验证集选优阶段权重
- `TuningRoundX/artifacts/history_val.csv`：验证集选优阶段训练日志
- `TuningRoundX/artifacts/best_meta_val.json`：验证集选优阶段元信息
- `TuningRoundX/artifacts/run_config_val.json`：验证集选优阶段完整配置
- `TuningRoundX/artifacts/best_model_full.npy`：full-data 重训权重
- `TuningRoundX/artifacts/history_full.csv`：full-data 重训日志
- `TuningRoundX/artifacts/best_meta_full.json`：full-data 重训元信息
- `TuningRoundX/artifacts/run_config_full.json`：full-data 重训完整配置
- `TuningRoundX/artifacts/split_indices.npz`：训练/验证划分索引
- `TuningRoundX/artifacts/search_log.csv`：网格搜索日志
- `TuningRoundX/artifacts/search_summary.json`：网格搜索摘要
- `TuningRoundX/figures/history.png`：训练曲线图（当前已按 `history_val.csv` 重绘，包含 val 曲线）
- `TuningRoundX/figures/confusion_count.png`：混淆矩阵（计数）
- `TuningRoundX/figures/confusion_norm.png`：混淆矩阵（归一化）
- `TuningRoundX/figures/confusion_matrix.csv`：混淆矩阵原始数据
- `TuningRoundX/figures/weight1_grid.png`：第一层隐藏层权重可视化
- `TuningRoundX/figures/error_cases.csv`：错例索引与预测详情
- `TuningRoundX/figures/errors/*.png`：错例图像

## 8. 模型选择语义解释

- 约定：
  - `best_model_val.npy`：验证阶段最优模型（用于超参数选择/训练过程分析）
  - `best_model_full.npy`：最终提交模型（用于最终测试与报告）
- 验证集阶段（`use_validation_split=True`）：
  - 训练会按 `val_acc` 进行早停与选优
  - 当 `val_acc` 在 `min_delta` 范围内并列时，使用更小的 `val_total_loss` 作为 tie-break
- 全量重训阶段（`use_validation_split=False`）：
  - 固定已选超参数后，在 `train+val` 上重训，不再进行验证集选优
  - 输出 `best_model_full.npy`（`final_model_full`）作为最终测试模型
