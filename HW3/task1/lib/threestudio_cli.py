"""Build threestudio launch.py CLI overrides."""

from __future__ import annotations

from pathlib import Path

from task1.lib.paths import (
    REPO_THREESTUDIO,
    SD15_PATH,
    SD21_PATH,
    THREESTUDIO_EXP_ROOT,
    ZERO123_PATH,
)
from task1.lib.runner import find_trial_dir, latest_ckpt, python_exe, run_cmd, threestudio_launch_base


def _q(value: str) -> str:
    """Hydra-style override with quoting for spaces."""
    if any(c in value for c in " \t"):
        return f'"{value}"'
    return value


def train_threestudio(
    *,
    config: str,
    name: str,
    tag: str,
    overrides: dict[str, str],
    gpu: int = 0,
    train: bool = True,
) -> Path:
    extra = [
        f"exp_root_dir={THREESTUDIO_EXP_ROOT.as_posix()}",
        f"name={name}",
        f"tag={tag}",
        "use_timestamp=False",
    ]
    for key, val in overrides.items():
        extra.append(f"{key}={_q(val)}")
    cmd = threestudio_launch_base(config=config, gpu=gpu, extra=extra)
    if train:
        cmd.insert(2, "--train")
    run_cmd(cmd, cwd=REPO_THREESTUDIO)
    return find_trial_dir(THREESTUDIO_EXP_ROOT, name, tag)


def export_mesh(
    trial_dir: Path,
    *,
    gpu: int = 0,
    exporter_type: str = "mesh-exporter",
    context_type: str = "cuda",
    extra: list[str] | None = None,
) -> None:
    parsed = trial_dir / "configs" / "parsed.yaml"
    ckpt = latest_ckpt(trial_dir)
    cmd = threestudio_launch_base(config=str(parsed), gpu=gpu, extra=[])
    cmd.insert(2, "--export")
    cmd.append(f"resume={ckpt.as_posix()}")
    cmd.append(f"system.exporter_type={exporter_type}")
    cmd.append(f"system.exporter.context_type={context_type}")
    if extra:
        cmd.extend(extra)
    run_cmd(cmd, cwd=REPO_THREESTUDIO)


def dreamfusion_overrides(
    *,
    prompt: str,
    negative_prompt: str | None,
    max_steps: int,
    sd_path: Path | None = None,
) -> dict[str, str]:
    sd = (sd_path or SD21_PATH).as_posix()
    out = {
        "system.prompt_processor.prompt": prompt,
        "system.prompt_processor.pretrained_model_name_or_path": sd,
        "system.guidance.pretrained_model_name_or_path": sd,
        "trainer.max_steps": str(max_steps),
    }
    if negative_prompt:
        out["system.prompt_processor.negative_prompt"] = negative_prompt
    return out


def magic123_coarse_overrides(
    *,
    image_path: Path,
    prompt: str,
    negative_prompt: str | None,
    max_steps: int,
    sd15_path: Path | None = None,
    zero123_path: Path | None = None,
) -> dict[str, str]:
    sd = (sd15_path or SD15_PATH).as_posix()
    z = (zero123_path or ZERO123_PATH).as_posix()
    out = {
        "data.image_path": image_path.as_posix(),
        "system.prompt_processor.prompt": prompt,
        "system.prompt_processor.pretrained_model_name_or_path": sd,
        "system.guidance.pretrained_model_name_or_path": sd,
        "system.guidance_3d.pretrained_model_name_or_path": z,
        "trainer.max_steps": str(max_steps),
    }
    if negative_prompt:
        out["system.prompt_processor.negative_prompt"] = negative_prompt
    return out


def magic123_refine_overrides(
    *,
    image_path: Path,
    prompt: str,
    negative_prompt: str | None,
    coarse_ckpt: Path,
    max_steps: int,
    sd15_path: Path | None = None,
    zero123_path: Path | None = None,
) -> dict[str, str]:
    out = magic123_coarse_overrides(
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        max_steps=max_steps,
        sd15_path=sd15_path,
        zero123_path=zero123_path,
    )
    out["system.geometry_convert_from"] = coarse_ckpt.as_posix()
    return out
