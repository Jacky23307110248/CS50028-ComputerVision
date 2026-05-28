#!/usr/bin/env bash
# Run on Linux (AMD). Windows may fail during video concat (PyAV/ffconcat).
set -euo pipefail
ROOT="${HW3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
python task2/scripts/convert_calvin_to_v30.py --env all "$@"
