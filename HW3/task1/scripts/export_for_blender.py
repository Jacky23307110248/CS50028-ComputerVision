#!/usr/bin/env python
"""Collect 2DGS + threestudio artifacts into fusion/export for local Blender (F1)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))

from task1.lib.paths import (  # noqa: E402
    OUTPUT_BACKGROUND,
    OUTPUT_FUSION,
    OUTPUT_OBJECT_A,
    OUTPUT_OBJECT_B,
    OUTPUT_OBJECT_C,
    ensure_output_dirs,
)


def _copy_glob(src_root: Path, pattern: str, dst: Path, prefix: str) -> list[str]:
    copied: list[str] = []
    if not src_root.is_dir():
        return copied
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_root.rglob(pattern)):
        if f.is_file():
            name = f"{prefix}_{f.name}" if prefix else f.name
            target = dst / name
            shutil.copy2(f, target)
            copied.append(str(target.relative_to(OUTPUT_FUSION)))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_FUSION / "export",
        help="Staged assets for Blender import",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    index: dict[str, list[str]] = {}

    # 2DGS point clouds (background + object A)
    for label, root, prefix in [
        ("background_garden", OUTPUT_BACKGROUND / "run", "bg"),
        ("object_a", OUTPUT_OBJECT_A / "run", "obj_a"),
    ]:
        pc_dir = root / "point_cloud"
        if pc_dir.is_dir():
            # iteration_* / point_cloud.ply
            index[label] = _copy_glob(pc_dir, "point_cloud.ply", out / "2dgs", prefix)

    # threestudio meshes already copied to fusion/blender by train scripts
    blender_src = OUTPUT_FUSION / "blender"
    if blender_src.is_dir():
        index["meshes"] = _copy_glob(blender_src, "*.obj", out / "meshes", "")
        index["meshes_mtl"] = _copy_glob(blender_src, "*.mtl", out / "meshes", "")

    # Copy manifests for future F2
    manifests: list[str] = []
    for root in (OUTPUT_BACKGROUND, OUTPUT_OBJECT_A, OUTPUT_OBJECT_B, OUTPUT_OBJECT_C):
        m = root / "artifacts" / "manifest.json"
        if m.is_file():
            dst = out / "manifests" / f"{root.name}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(m, dst)
            manifests.append(str(dst.relative_to(OUTPUT_FUSION)))
        # nested stage manifests
        art = root / "artifacts"
        if art.is_dir():
            for mf in art.rglob("manifest.json"):
                rel = mf.relative_to(root)
                dst = out / "manifests" / root.name / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mf, dst)
                manifests.append(str(dst.relative_to(OUTPUT_FUSION)))
    index["manifests"] = manifests

    readme = OUTPUT_FUSION / "BLENDER_F1.md"
    readme.write_text(
        """# Blender 融合 (F1)

## 导入资产

1. **背景 / 物体 A (2DGS)**：`export/2dgs/` 下的 `*.ply` 点云。可用 Blender 插件或先转为 mesh/实例化显示。
2. **物体 B / C (AIGC mesh)**：`export/meshes/` 下的 `object_b_*.obj`、`object_c_*.obj`（含 mtl）。

## 建议流程

1. 以 garden 背景尺度为参考，统一单位（threestudio 物体约在 2m 半径球内，需手动缩放对齐）。
2. 将 B/C mesh 放入背景合适位置；A 的 2DGS 可单独一层或烘焙为 mesh 后合并。
3. 添加相机路径（圆环或关键帧），渲染多视角漫游 MP4。

## 完整 GPU 产物

若需日后 F2 代码级融合，原始训练目录保存在：

- `task1/outputs/background_garden/artifacts/`
- `task1/outputs/object_a/artifacts/`
- `task1/outputs/object_b/artifacts/`（含 ckpt / parsed.yaml）
- `task1/outputs/object_c/artifacts/`（coarse + refine）

`export/manifests/` 为索引副本。
""",
        encoding="utf-8",
    )

    index_path = out / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Export index: {index_path}")
    print(f"Blender guide: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
