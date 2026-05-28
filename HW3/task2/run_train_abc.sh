#!/usr/bin/env bash
set -euo pipefail
ROOT="${HW3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
python task2/scripts/build_splits.py
python task2/scripts/train_abc.py "$@"
