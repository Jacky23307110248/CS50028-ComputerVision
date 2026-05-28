#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 <checkpoint_pretrained_model_dir> [extra args to eval_zero_shot_d.py]"
  exit 1
fi
CKPT="$1"
shift
ROOT="${HW3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
python task2/scripts/eval_zero_shot_d.py --checkpoint "$CKPT" "$@"
