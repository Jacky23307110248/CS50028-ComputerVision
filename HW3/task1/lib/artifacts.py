"""Snapshot GPU training outputs for Blender (F1) and future F2 automation."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task1.lib.runner import latest_ckpt


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_manifest(
    dest: Path,
    *,
    kind: str,
    source_dirs: list[Path],
    extra: dict[str, Any] | None = None,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [str(p) for p in source_dirs],
        "files": [],
    }
    if extra:
        manifest.update(extra)

    for src_root in source_dirs:
        if not src_root.exists():
            continue
        if src_root.is_file():
            manifest["files"].append({"path": str(src_root), "role": "file"})
            continue
        for f in sorted(src_root.rglob("*")):
            if f.is_file():
                manifest["files"].append(
                    {
                        "path": _rel(f, src_root),
                        "role": "artifact",
                        "root": str(src_root),
                        "size_bytes": f.stat().st_size,
                    }
                )

    out = dest / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def snapshot_dir(src: Path, dst: Path) -> None:
    """Copy directory tree into artifacts/full_run (overwrite)."""
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def snapshot_threestudio_trial(
    trial_dir: Path,
    output_root: Path,
    *,
    stage: str,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> Path:
    """Save ckpt, configs, saves, tb for one threestudio stage."""
    artifacts = output_root / "artifacts" / stage
    snapshot_dir(trial_dir, artifacts / "trial")
    extra: dict[str, Any] = {
        "stage": stage,
        "checkpoint": str(latest_ckpt(trial_dir)),
        "parsed_config": str(trial_dir / "configs" / "parsed.yaml"),
    }
    if prompt:
        extra["prompt"] = prompt
    if negative_prompt:
        extra["negative_prompt"] = negative_prompt
    write_manifest(artifacts, kind="threestudio", source_dirs=[trial_dir], extra=extra)
    return artifacts


def snapshot_2dgs_run(model_path: Path, output_root: Path) -> Path:
    artifacts = output_root / "artifacts" / "2dgs"
    snapshot_dir(model_path, artifacts / "model")
    extra = {
        "point_cloud": str(model_path / "point_cloud"),
        "cfg_args": str(model_path / "cfg_args"),
    }
    write_manifest(
        artifacts,
        kind="2dgs",
        source_dirs=[model_path],
        extra=extra,
    )
    return artifacts


def copy_export_meshes(trial_dir: Path, blender_dir: Path, prefix: str) -> list[Path]:
    """Copy threestudio export meshes (save/it*-export/*.obj) into fusion/blender."""
    blender_dir.mkdir(parents=True, exist_ok=True)
    search_roots = [trial_dir / "save", trial_dir]
    copied: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for mesh in sorted(root.rglob("*.obj")):
            if mesh in seen:
                continue
            seen.add(mesh)
            dst = blender_dir / f"{prefix}_{mesh.name}"
            shutil.copy2(mesh, dst)
            copied.append(dst)
            mtl = mesh.with_suffix(".mtl")
            if mtl.is_file():
                shutil.copy2(mtl, blender_dir / f"{prefix}_{mtl.name}")
            for tex in mesh.parent.glob(f"{mesh.stem}*"):
                if tex.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(tex, blender_dir / f"{prefix}_{tex.name}")
    return copied
