#!/usr/bin/env python
"""Verify HW3 task1 data, repos, weights, and DSW venv hints."""

from __future__ import annotations

import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task1.lib.paths import (  # noqa: E402
    CACHE_HF,
    HW3_ROOT,
    OBJECT_A_ROOT,
    OBJECT_B_ROOT,
    OBJECT_C_ROOT,
    REPO_2DGS,
    REPO_THREESTUDIO,
    SD15_PATH,
    SD21_PATH,
    VENV_2DGS,
    VENV_THREESTUDIO,
    ZERO123_PATH,
    garden_scene_root,
    venv_python,
)
from task1.lib.prompts import PromptFiles  # noqa: E402


def _ok(path: Path) -> bool:
    return path.exists()


def main() -> int:
    print(f"HW3_ROOT={HW3_ROOT}")
    ok = True

    for label, path in [
        ("2DGS repo", REPO_2DGS),
        ("threestudio repo", REPO_THREESTUDIO),
        ("SD 1.5", SD15_PATH),
        ("SD 2.1", SD21_PATH),
        ("Zero123 diffusers", ZERO123_PATH),
    ]:
        exists = _ok(path)
        print(f"  {label}: {path} -> {'OK' if exists else 'MISSING'}")
        ok = ok and exists

    garden = garden_scene_root()
    garden_ok = (garden / "images").is_dir()
    print(f"  garden scene: {garden} -> {'OK' if garden_ok else 'MISSING images/'}")
    ok = ok and garden_ok

    convert_py = REPO_2DGS / "convert.py"
    if convert_py.is_file():
        text = convert_py.read_text(encoding="utf-8")
        if "FeatureExtraction.use_gpu" in text:
            print("  convert.py COLMAP 3.13 patch: OK")
        else:
            print("  convert.py COLMAP 3.13 patch: MISSING (need patched repos)")
            ok = False

    raw_video = OBJECT_A_ROOT / "raw" / "input.mp4"
    colmap_input = OBJECT_A_ROOT / "input"
    print(f"  object_A raw video: {raw_video} -> {'OK' if raw_video.is_file() else 'optional'}")
    print(
        f"  object_A COLMAP input/: {colmap_input} -> "
        f"{'OK' if colmap_input.is_dir() else 'run prepare_object_a.py after C1 reshoot'}"
    )

    for name, root in [("object_B", OBJECT_B_ROOT), ("object_C", OBJECT_C_ROOT)]:
        pf = PromptFiles(root)
        if pf.prompt_path.is_file():
            try:
                p = pf.require_prompt()
                print(f"  {name} prompt.txt: OK ({p[:60]}{'...' if len(p) > 60 else ''})")
            except ValueError as exc:
                print(f"  {name} prompt.txt: INVALID ({exc})")
                ok = False
        else:
            print(f"  {name} prompt.txt: MISSING (required before train)")

        img_c = root / "rgba.png"
        if name == "object_C":
            print(f"  object_C rgba.png: {'OK' if img_c.is_file() else 'MISSING'}")

    print("\nDSW venvs (no conda — run setup_dsw_env.py if missing):")
    for label, venv, which in [
        ("2DGS", VENV_2DGS, "2dgs"),
        ("threestudio", VENV_THREESTUDIO, "ts"),
    ]:
        py = venv_python(which)
        if py.is_file():
            print(f"  {label}: {venv} -> OK ({py})")
        else:
            print(f"  {label}: {venv} -> MISSING (python task1/scripts/setup_dsw_env.py --target {which})")
    print("\nModelScope sync (HW3 root):")
    print("  python upload_task1_to_workspace.py --mode assets")
    print("  python download_task1_on_dsw.py --mode assets --only-missing")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
