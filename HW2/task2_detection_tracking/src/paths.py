"""工程内统一路径（相对 HW2 根目录下的 evilspirit05 与 task2 子目录）。"""

from pathlib import Path

TASK2_ROOT = Path(__file__).resolve().parents[1]
HW2_ROOT = TASK2_ROOT.parent

EVILSPIRIT14 = HW2_ROOT / "evilspirit05" / "visdrone" / "versions" / "14"

DETECT_VIDEO_DIR = TASK2_ROOT / "detectVideo"
DEFAULT_INPUT_VIDEO = DETECT_VIDEO_DIR / "input.mp4"
DEFAULT_OUTPUT_VIDEO = DETECT_VIDEO_DIR / "output.mp4"
DEFAULT_TRACK_LOG = DETECT_VIDEO_DIR / "track_log.csv"

CONFIGS_DIR = TASK2_ROOT / "configs"
VISDRONE_DATA_YAML = CONFIGS_DIR / "visdrone_local.yaml"

WEIGHTS_DIR = TASK2_ROOT / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"
