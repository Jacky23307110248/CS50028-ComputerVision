#!/usr/bin/env python
"""Create task1 venvs on DSW (no conda): reuse system torch via --system-site-packages."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task1.lib.paths import (  # noqa: E402
    HW3_ROOT,
    REPO_2DGS,
    REPO_THREESTUDIO,
    TASK1_ROOT,
    VENV_2DGS,
    VENV_THREESTUDIO,
    venv_python,
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _venv_bin(venv_root: Path) -> Path:
    return venv_python("2dgs" if venv_root == VENV_2DGS else "ts")


def create_venv(venv_root: Path, *, recreate: bool) -> Path:
    if recreate and venv_root.exists():
        print(f"Removing old venv: {venv_root}")
        shutil.rmtree(venv_root)
    if not venv_root.exists():
        # 继承 DSW 预装的 torch / torchvision / cuda 绑定
        _run(
            [
                sys.executable,
                "-m",
                "venv",
                "--system-site-packages",
                str(venv_root),
            ]
        )
    py = _venv_bin(venv_root)
    _run([str(py), "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    return py


def setup_2dgs(*, recreate: bool) -> None:
    py = create_venv(VENV_2DGS, recreate=recreate)
    _run([str(py), "-m", "pip", "install", "-r", str(TASK1_ROOT / "requirements_2dgs.txt")])
    for sub in ("diff-surfel-rasterization", "simple-knn"):
        sub_path = REPO_2DGS / "submodules" / sub
        if not sub_path.is_dir():
            raise FileNotFoundError(f"Missing submodule: {sub_path}")
        _run([str(py), "-m", "pip", "install", str(sub_path)], cwd=REPO_2DGS)
    # smoke import
    _run(
        [
            str(py),
            "-c",
            "import torch; import diff_surfel_rasterization; print('2dgs ok', torch.__version__, torch.cuda.is_available())",
        ]
    )


def setup_threestudio(*, recreate: bool) -> None:
    py = create_venv(VENV_THREESTUDIO, recreate=recreate)
    req = REPO_THREESTUDIO / "requirements.txt"
    if not req.is_file():
        raise FileNotFoundError(req)
    _run([str(py), "-m", "pip", "install", "-r", str(req)])
    _run([str(py), "-m", "pip", "install", "-r", str(TASK1_ROOT / "requirements_threestudio.txt")])
    _run(
        [
            str(py),
            "-c",
            "import torch; import lightning; import diffusers; print('threestudio deps ok', torch.__version__, torch.cuda.is_available())",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup task1 venvs on DSW (no conda).")
    parser.add_argument(
        "--target",
        choices=("2dgs", "ts", "all"),
        default="all",
        help="2dgs=background/object_A, ts=object_B/C (threestudio)",
    )
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild venv(s)")
    args = parser.parse_args()

    print(f"HW3_ROOT={HW3_ROOT}")
    print(f"Base Python: {sys.executable}")

    try:
        import torch

        print(f"System torch: {torch.__version__}, cuda={torch.cuda.is_available()}")
    except ImportError:
        print("[warn] System torch not importable; venv may need manual torch install.")

    if args.target in ("2dgs", "all"):
        setup_2dgs(recreate=args.recreate)
        print(f"\n2DGS venv ready: {VENV_2DGS}")
        print(f"  source {VENV_2DGS}/bin/activate")

    if args.target in ("ts", "all"):
        setup_threestudio(recreate=args.recreate)
        print(f"\nthreestudio venv ready: {VENV_THREESTUDIO}")
        print(f"  source {VENV_THREESTUDIO}/bin/activate")

    print("\nTrain (after activate):")
    print("  python task1/scripts/run_train_background.py")
    print("  python task1/scripts/run_train_object_b.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
