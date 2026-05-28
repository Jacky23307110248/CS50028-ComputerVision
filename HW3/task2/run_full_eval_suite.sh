#!/usr/bin/env bash
# Full eval suite (final checkpoint 100k only). Run on AMD after training.
set -euo pipefail
ROOT="${HW3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

B_CKPT="task2/outputs/act_B_only/checkpoints/0100000/pretrained_model"
ABC_CKPT="task2/outputs/act_ABC_mixed/checkpoints/0100000/pretrained_model"

python task2/scripts/eval_zero_shot_d.py --checkpoint "$B_CKPT" --split d_test --run-name eval_D_B_only --model-tag act_B_only
python task2/scripts/eval_zero_shot_d.py --checkpoint "$ABC_CKPT" --split d_test --run-name eval_D_ABC --model-tag act_ABC_mixed

python task2/scripts/eval_zero_shot_d.py --checkpoint "$B_CKPT" --split b_val --run-name eval_B_val_B_only --model-tag act_B_only
python task2/scripts/eval_zero_shot_d.py --checkpoint "$ABC_CKPT" --split abc_val --run-name eval_ABC_val_ABC --model-tag act_ABC_mixed

echo "All eval runs finished. See task2/outputs/eval_*/eval_summary.json"
