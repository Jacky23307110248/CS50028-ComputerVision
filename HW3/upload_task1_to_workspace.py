# upload_task1_to_workspace.py
# 本地 Windows 运行：上传到魔搭数据集 sSzHox/task1_Data
# 上传完成后，在 Task1 DSW 实例运行 download_task1_on_dsw.py
#
# 用法:
#   set MODELSCOPE_API_TOKEN=ms-xxxx
#   python upload_task1_to_workspace.py --mode assets          # SD2.1 精简上传（默认）
#   python upload_task1_to_workspace.py --mode assets --sd21-full  # SD2.1 整卷 ~29GB
#   python upload_task1_to_workspace.py --mode full
#   python upload_task1_to_workspace.py --mode object_a
#   python upload_task1_to_workspace.py --mode object_b
#   python upload_task1_to_workspace.py --mode object_c
#
# --mode 说明:
#   assets    : garden + 权重 + repos（不含 object_A / object_B / object_C）
#   full      : 上述全部 + object_A + object_B + object_C（缺或校验失败则报错）
#   object_a  : 仅 object_A
#   object_b  : 仅 object_B（data/object_B/prompt.txt）
#   object_c  : 仅 object_C

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from modelscope.hub.api import HubApi

REPO_ID = "sSzHox/task1_Data"
PATH_IN_REPO = "hw3-task1"

HW3_ROOT = Path(r"D:\大三下\计算机视觉\HW3")

IGNORE = [
    "**/.git/**",
    "**/.cache/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
]
MAX_WORKERS = 4

# SD2.1 精简：只保留 DreamFusion/threestudio 需要的 diffusers safetensors + 配置（~5GB）
SD21_SLIM_EXTRA_IGNORE = [
    "*.ckpt",
    "**/*.bin",
    "**/*fp16*",
    "v2-1_512-*",
]

EXPECTED_REMOTES = {
    "repos-2dgs": "https://github.com/hbb1/2d-gaussian-splatting.git",
    "repos-threestudio": "https://github.com/threestudio-project/threestudio.git",
}


@dataclass(frozen=True)
class Shard:
    name: str
    local_dir: Path
    repo_subpath: str


def get_token() -> str:
    token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get(
        "MODELSCOPE_ACCESS_TOKEN"
    )
    if not token:
        print("请设置环境变量 MODELSCOPE_API_TOKEN（不要写进代码/不要提交 git）")
        print("  CMD:        set MODELSCOPE_API_TOKEN=ms-xxxx")
        print('  PowerShell: $env:MODELSCOPE_API_TOKEN = "ms-xxxx"')
        sys.exit(1)
    return token


def shard_definitions() -> dict[str, Shard]:
    return {
        "garden": Shard(
            name="garden",
            local_dir=HW3_ROOT / "data" / "mipnerf360" / "_download" / "360_v2" / "garden",
            repo_subpath="garden",
        ),
        "weights-sd21": Shard(
            name="weights-sd21",
            local_dir=HW3_ROOT / "cache" / "hf" / "stable-diffusion-2-1-base",
            repo_subpath="weights-sd21",
        ),
        "weights-sd15": Shard(
            name="weights-sd15",
            local_dir=HW3_ROOT / "cache" / "hf" / "stable-diffusion-v1-5",
            repo_subpath="weights-sd15",
        ),
        "weights-zero123": Shard(
            name="weights-zero123",
            local_dir=HW3_ROOT / "cache" / "hf" / "zero123-diffusers",
            repo_subpath="weights-zero123",
        ),
        "repos-2dgs": Shard(
            name="repos-2dgs",
            local_dir=HW3_ROOT / "repos" / "2d-gaussian-splatting",
            repo_subpath="repos-2dgs",
        ),
        "repos-threestudio": Shard(
            name="repos-threestudio",
            local_dir=HW3_ROOT / "repos" / "threestudio",
            repo_subpath="repos-threestudio",
        ),
        "object_a": Shard(
            name="object_a",
            local_dir=HW3_ROOT / "data" / "object_A",
            repo_subpath="object_a",
        ),
        "object_b": Shard(
            name="object_b",
            local_dir=HW3_ROOT / "data" / "object_B",
            repo_subpath="object_b",
        ),
        "object_c": Shard(
            name="object_c",
            local_dir=HW3_ROOT / "data" / "object_C",
            repo_subpath="object_c",
        ),
    }


def _count_files(folder: Path, pattern: str = "*") -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.glob(pattern) if p.is_file())


def _read_git_remote(repo_dir: Path) -> str | None:
    cfg = repo_dir / ".git" / "config"
    if not cfg.is_file():
        return None
    text = cfg.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("url ="):
            return line.split("=", 1)[1].strip()
    return None


def validate_garden(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 garden 目录: {shard.local_dir}")
    images = shard.local_dir / "images"
    sparse0 = shard.local_dir / "sparse" / "0"
    n_img = _count_files(images, "*.jpg") + _count_files(images, "*.JPG")
    if n_img < 50:
        raise FileNotFoundError(
            f"garden/images 图片过少 ({n_img} 张，期望 >=50): {images}"
        )
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse0 / name).is_file():
            raise FileNotFoundError(f"缺少 garden COLMAP 文件: {sparse0 / name}")
    if not (shard.local_dir / "poses_bounds.npy").is_file():
        raise FileNotFoundError(f"缺少 poses_bounds.npy: {shard.local_dir}")


def validate_weights_sd21(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 SD2.1 目录: {shard.local_dir}")
    unet = shard.local_dir / "unet" / "diffusion_pytorch_model.safetensors"
    vae = shard.local_dir / "vae" / "diffusion_pytorch_model.safetensors"
    te = shard.local_dir / "text_encoder" / "model.safetensors"
    for p in (unet, vae, te):
        if not p.is_file() or p.stat().st_size < 50 * 1024 * 1024:
            raise FileNotFoundError(f"SD2.1 权重不完整: {p}")
    if not (shard.local_dir / "model_index.json").is_file():
        raise FileNotFoundError(f"缺少 model_index.json: {shard.local_dir}")


def validate_weights_sd15(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 SD1.5 目录: {shard.local_dir}")
    unet = shard.local_dir / "unet" / "diffusion_pytorch_model.safetensors"
    vae = shard.local_dir / "vae" / "diffusion_pytorch_model.safetensors"
    te = shard.local_dir / "text_encoder" / "model.safetensors"
    for p in (unet, vae, te):
        if not p.is_file() or p.stat().st_size < 50 * 1024 * 1024:
            raise FileNotFoundError(f"SD1.5 权重不完整: {p}")


def validate_weights_zero123(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 Zero123 目录: {shard.local_dir}")
    unet = shard.local_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    vae = shard.local_dir / "vae" / "diffusion_pytorch_model.fp16.safetensors"
    ie = shard.local_dir / "image_encoder" / "model.fp16.safetensors"
    for p in (unet, vae, ie):
        if not p.is_file() or p.stat().st_size < 50 * 1024 * 1024:
            raise FileNotFoundError(f"Zero123 fp16 权重不完整: {p}")


def validate_repos_2dgs(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 2DGS 仓库: {shard.local_dir}")
    remote = _read_git_remote(shard.local_dir)
    expected = EXPECTED_REMOTES["repos-2dgs"]
    if remote != expected:
        raise ValueError(f"2DGS remote 不符合预期: {remote!r} (期望 {expected})")
    for rel in (
        "train.py",
        "convert.py",
        "submodules/diff-surfel-rasterization/setup.py",
    ):
        if not (shard.local_dir / rel).is_file():
            raise FileNotFoundError(f"2DGS 仓库缺少关键文件: {rel}")


def validate_repos_threestudio(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 threestudio 仓库: {shard.local_dir}")
    remote = _read_git_remote(shard.local_dir)
    expected = EXPECTED_REMOTES["repos-threestudio"]
    if remote != expected:
        raise ValueError(f"threestudio remote 不符合预期: {remote!r} (期望 {expected})")
    for rel in (
        "launch.py",
        "configs/dreamfusion-sd.yaml",
        "configs/magic123-coarse-sd.yaml",
    ):
        if not (shard.local_dir / rel).is_file():
            raise FileNotFoundError(f"threestudio 仓库缺少关键文件: {rel}")


def validate_object_a(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 object_A 目录: {shard.local_dir}")
    images = shard.local_dir / "images"
    sparse0 = shard.local_dir / "sparse" / "0"
    n_img = _count_files(images, "*.jpg") + _count_files(images, "*.JPG")
    if n_img < 10:
        raise FileNotFoundError(
            f"object_A 尚未完成 COLMAP（images 仅 {n_img} 张，期望 >=10）: {images}\n"
            "仅有 raw/input.mp4 不够，请先抽帧并跑 convert.py。"
        )
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse0 / name).is_file():
            raise FileNotFoundError(f"缺少 object_A sparse 文件: {sparse0 / name}")


def validate_object_b(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 object_B 目录: {shard.local_dir}")
    prompt = shard.local_dir / "prompt.txt"
    if not prompt.is_file():
        raise FileNotFoundError(
            f"缺少 object_B 文本 Prompt: {prompt}\n"
            "请创建 data/object_B/prompt.txt 并写入 DreamFusion 描述。"
        )
    text = prompt.read_text(encoding="utf-8").strip()
    if not text:
        raise FileNotFoundError(f"object_B/prompt.txt 为空: {prompt}")


def validate_object_c(shard: Shard) -> None:
    if not shard.local_dir.is_dir():
        raise FileNotFoundError(f"缺少 object_C 目录: {shard.local_dir}")
    exts = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
    images = [p for p in shard.local_dir.rglob("*") if p.is_file() and p.suffix in exts]
    if not images:
        raise FileNotFoundError(
            f"object_C 中没有前景图片: {shard.local_dir}\n"
            "请至少放入一张去背景后的物体图（jpg/png）。"
        )


VALIDATORS = {
    "garden": validate_garden,
    "weights-sd21": validate_weights_sd21,
    "weights-sd15": validate_weights_sd15,
    "weights-zero123": validate_weights_zero123,
    "repos-2dgs": validate_repos_2dgs,
    "repos-threestudio": validate_repos_threestudio,
    "object_a": validate_object_a,
    "object_b": validate_object_b,
    "object_c": validate_object_c,
}

ASSET_SHARDS = [
    "garden",
    "weights-sd21",
    "weights-sd15",
    "weights-zero123",
    "repos-2dgs",
    "repos-threestudio",
]
OBJECT_SHARDS = ["object_a", "object_b", "object_c"]


def shards_for_mode(mode: str) -> list[str]:
    if mode == "assets":
        return list(ASSET_SHARDS)
    if mode == "full":
        return ASSET_SHARDS + OBJECT_SHARDS
    if mode == "object_a":
        return ["object_a"]
    if mode == "object_b":
        return ["object_b"]
    if mode == "object_c":
        return ["object_c"]
    raise ValueError(f"未知 mode: {mode}")


def validate_shards(shard_names: list[str], shards: dict[str, Shard]) -> None:
    for name in shard_names:
        print(f"校验 {name} …")
        VALIDATORS[name](shards[name])
        print(f"  OK: {shards[name].local_dir}")


def upload_shards(
    api: HubApi,
    shard_names: list[str],
    shards: dict[str, Shard],
    *,
    sd21_full: bool = False,
) -> None:
    print(f"目标数据集: {REPO_ID}")
    print(f"仓库内根路径: {PATH_IN_REPO}/")
    print(f"将上传 {len(shard_names)} 卷: {shard_names}")
    if "weights-sd21" in shard_names:
        print(
            "SD2.1 上传模式:",
            "完整 ~29GB" if sd21_full else "精简 ~5GB（仅 diffusers safetensors + 配置）",
        )
    print()

    for i, name in enumerate(shard_names, 1):
        shard = shards[name]
        target = f"{PATH_IN_REPO}/{shard.repo_subpath}"
        ignore = list(IGNORE)
        if name == "weights-sd21" and not sd21_full:
            ignore = ignore + SD21_SLIM_EXTRA_IGNORE
        print(f"[{i}/{len(shard_names)}] 上传 {name} -> {target}/")
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=str(shard.local_dir),
            path_in_repo=target,
            repo_type="dataset",
            ignore_patterns=ignore,
            max_workers=MAX_WORKERS,
            commit_message=f"Upload Task1 shard {name} for HW3",
        )
        print(f"[{i}/{len(shard_names)}] 完成: {name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload HW3 Task1 assets to ModelScope")
    parser.add_argument(
        "--mode",
        choices=["assets", "full", "object_a", "object_b", "object_c"],
        default="assets",
        help="assets=不含物体; full=全部; object_a/object_b/object_c=单独补传",
    )
    parser.add_argument(
        "--sd21-full",
        action="store_true",
        help="SD2.1 上传完整文件夹（~29GB，含冗余 ckpt/bin/fp16）；默认精简 ~5GB",
    )
    args = parser.parse_args()

    shards = shard_definitions()
    selected = shards_for_mode(args.mode)
    validate_shards(selected, shards)

    token = get_token()
    api = HubApi()
    api.login(token)

    upload_shards(api, selected, shards, sd21_full=args.sd21_full)
    print("全部上传完成。")
    print("请到 Task1 DSW 实例运行: python download_task1_on_dsw.py")


if __name__ == "__main__":
    main()
