"""在 Task1 PAI-DSW 实例运行: python download_task1_on_dsw.py

与 Task2 共用根目录 /mnt/workspace/HW3，但题目一数据落在 cache/、data/mipnerf360/、repos/ 等子路径。

用法:
  export MODELSCOPE_API_TOKEN=ms-xxxx
  python download_task1_on_dsw.py --mode assets
  python download_task1_on_dsw.py --mode assets --only-missing
  python download_task1_on_dsw.py --shard garden
  python download_task1_on_dsw.py --shard sd21
  python download_task1_on_dsw.py --mode object_a
  python download_task1_on_dsw.py --mode object_b

如何在魔搭免费 GPU 实例确认 Token / 账号（问题 4）:
  1) 浏览器打开 https://www.modelscope.cn 并登录（与创建 sSzHox/hw3-task1-data 同一账号）
  2) 右上角头像 -> 个人中心 -> API Token：复制 ms- 开头 Token
  3) DSW Terminal 执行:
       echo $MODELSCOPE_API_TOKEN
     若为空则:
       export MODELSCOPE_API_TOKEN=ms-xxxx
       export MODELSCOPE_ACCESS_TOKEN=$MODELSCOPE_API_TOKEN
  4) 数据集页打开 sSzHox/task1_Data，确认「所有者」是你当前登录账号
  5) 若数据集为私有，必须用上述 Token；免费实例与 Task2 实例只要同一魔搭账号+Token 即可访问同一数据集
  6) 测试:
       python -c "from modelscope.hub.api import HubApi; HubApi().login('$MODELSCOPE_API_TOKEN'); print('login ok')"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

REPO_ID = "sSzHox/task1_Data"
PATH_IN_REPO = "hw3-task1"
HW3_ROOT = Path("/mnt/workspace/HW3")

ASSET_SHARDS = [
    "garden",
    "weights-sd21",
    "weights-sd15",
    "weights-zero123",
    "repos-2dgs",
    "repos-threestudio",
]
OBJECT_SHARDS = ["object_a", "object_b", "object_c"]
SHARD_NAMES = ASSET_SHARDS + OBJECT_SHARDS

SHARD_ALIASES = {
    "garden": "garden",
    "sd21": "weights-sd21",
    "sd15": "weights-sd15",
    "zero123": "weights-zero123",
    "2dgs": "repos-2dgs",
    "threestudio": "repos-threestudio",
    "object_a": "object_a",
    "object_b": "object_b",
    "object_c": "object_c",
    "a": "object_a",
    "b": "object_b",
    "c": "object_c",
}


@dataclass(frozen=True)
class LayoutTarget:
    shard: str
    repo_subpath: str
    local_dir: Path


def layout_targets() -> dict[str, LayoutTarget]:
    return {
        "garden": LayoutTarget(
            "garden", "garden", HW3_ROOT / "data" / "mipnerf360" / "garden"
        ),
        "weights-sd21": LayoutTarget(
            "weights-sd21",
            "weights-sd21",
            HW3_ROOT / "cache" / "hf" / "stable-diffusion-2-1-base",
        ),
        "weights-sd15": LayoutTarget(
            "weights-sd15",
            "weights-sd15",
            HW3_ROOT / "cache" / "hf" / "stable-diffusion-v1-5",
        ),
        "weights-zero123": LayoutTarget(
            "weights-zero123",
            "weights-zero123",
            HW3_ROOT / "cache" / "hf" / "zero123-diffusers",
        ),
        "repos-2dgs": LayoutTarget(
            "repos-2dgs",
            "repos-2dgs",
            HW3_ROOT / "repos" / "2d-gaussian-splatting",
        ),
        "repos-threestudio": LayoutTarget(
            "repos-threestudio",
            "repos-threestudio",
            HW3_ROOT / "repos" / "threestudio",
        ),
        "object_a": LayoutTarget("object_a", "object_a", HW3_ROOT / "data" / "object_A"),
        "object_b": LayoutTarget("object_b", "object_b", HW3_ROOT / "data" / "object_B"),
        "object_c": LayoutTarget("object_c", "object_c", HW3_ROOT / "data" / "object_C"),
    }


def get_token() -> str:
    token = os.environ.get("MODELSCOPE_ACCESS_TOKEN") or os.environ.get(
        "MODELSCOPE_API_TOKEN"
    )
    if not token:
        raise SystemExit(
            "未设置 Token。私有数据集必须：\n"
            "  export MODELSCOPE_API_TOKEN=ms-xxxx\n"
            "  export MODELSCOPE_ACCESS_TOKEN=$MODELSCOPE_API_TOKEN"
        )
    return token


def login(token: str) -> None:
    HubApi().login(token)
    print("ModelScope 已登录")


def shards_for_mode(mode: str) -> list[str]:
    if mode == "assets":
        return list(ASSET_SHARDS)
    if mode == "full":
        return list(SHARD_NAMES)
    if mode == "object_a":
        return ["object_a"]
    if mode == "object_b":
        return ["object_b"]
    if mode == "object_c":
        return ["object_c"]
    raise ValueError(f"未知 mode: {mode}")


def resolve_shard(name: str) -> str:
    if name in SHARD_NAMES:
        return name
    if name in SHARD_ALIASES:
        return SHARD_ALIASES[name]
    raise ValueError(f"未知 shard: {name}，可选: {sorted(SHARD_ALIASES)}")


def staging_root() -> Path:
    return HW3_ROOT / "_download" / "task1_modelscope"


def download_shard(shard: str, token: str) -> None:
    target = layout_targets()[shard]
    staging = staging_root()
    staging.mkdir(parents=True, exist_ok=True)

    patterns = [
        f"{PATH_IN_REPO}/{target.repo_subpath}/**",
        f"{target.repo_subpath}/**",
    ]
    for pattern in patterns:
        print(f"拉取 {shard}: {pattern}")
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            local_dir=str(staging),
            allow_patterns=[pattern],
            token=token,
        )
        normalize_layout()
        if target.local_dir.exists() and _shard_present(shard):
            print(f"  已成功: {target.local_dir}")
            return
    print(f"  警告: {shard} 下载后仍未通过校验，请检查 Token 或魔搭路径。")


def _find_downloaded_dir(shard: str, repo_subpath: str) -> Path | None:
    staging = staging_root()
    candidates = [
        staging / PATH_IN_REPO / repo_subpath,
        staging / repo_subpath,
        staging / PATH_IN_REPO / repo_subpath.split("/")[-1],
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


def normalize_layout() -> None:
    """把 staging 中的分卷挪到 HW3 正式目录。"""
    for shard, target in layout_targets().items():
        src = _find_downloaded_dir(shard, target.repo_subpath)
        if src is None:
            continue
        target.local_dir.parent.mkdir(parents=True, exist_ok=True)
        if target.local_dir.exists():
            shutil.rmtree(target.local_dir)
        print(f"整理: {src} -> {target.local_dir}")
        shutil.move(str(src), str(target.local_dir))


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix in exts)


def verify_shard(shard: str) -> list[str]:
    errors: list[str] = []
    target = layout_targets()[shard]
    p = target.local_dir

    if shard == "garden":
        n = _count_images(p / "images")
        if n < 50:
            errors.append(f"garden/images 仅 {n} 张 (期望>=50)")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            if not (p / "sparse" / "0" / name).is_file():
                errors.append(f"缺少 {name}")
    elif shard == "weights-sd21":
        for rel in (
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
        ):
            if not (p / rel).is_file():
                errors.append(f"缺少 {rel}")
    elif shard == "weights-sd15":
        for rel in (
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
        ):
            if not (p / rel).is_file():
                errors.append(f"缺少 {rel}")
    elif shard == "weights-zero123":
        for rel in (
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "image_encoder/model.fp16.safetensors",
        ):
            if not (p / rel).is_file():
                errors.append(f"缺少 {rel}")
    elif shard == "repos-2dgs":
        for rel in ("train.py", "convert.py"):
            if not (p / rel).is_file():
                errors.append(f"缺少 {rel}")
    elif shard == "repos-threestudio":
        for rel in ("launch.py", "configs/dreamfusion-sd.yaml"):
            if not (p / rel).is_file():
                errors.append(f"缺少 {rel}")
    elif shard == "object_a":
        n = _count_images(p / "images")
        if n < 10:
            errors.append(f"object_A/images 仅 {n} 张 (期望>=10)")
    elif shard == "object_b":
        prompt = p / "prompt.txt"
        if not prompt.is_file():
            errors.append("缺少 prompt.txt")
        elif not prompt.read_text(encoding="utf-8").strip():
            errors.append("prompt.txt 为空")
    elif shard == "object_c":
        exts = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
        imgs = [x for x in p.rglob("*") if x.is_file() and x.suffix in exts] if p.is_dir() else []
        if not imgs:
            errors.append("object_C 无图片")

    return errors


def _shard_present(shard: str) -> bool:
    return not verify_shard(shard)


def missing_shards(shard_names: list[str]) -> list[str]:
    return [s for s in shard_names if not _shard_present(s)]


def verify_all(shard_names: list[str]) -> None:
    failed: dict[str, list[str]] = {}
    for shard in shard_names:
        errs = verify_shard(shard)
        if errs:
            failed[shard] = errs
        else:
            print(f"  OK {shard}: {layout_targets()[shard].local_dir}")
    if failed:
        msg = ["校验失败:"]
        for shard, errs in failed.items():
            msg.append(f"  [{shard}]")
            msg.extend(f"    - {e}" for e in errs)
        raise SystemExit("\n".join(msg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HW3 Task1 assets on DSW")
    parser.add_argument(
        "--mode",
        choices=["assets", "full", "object_a", "object_b", "object_c"],
        default="assets",
    )
    parser.add_argument(
        "--shard",
        choices=sorted(SHARD_ALIASES.keys()),
        help="仅下载指定卷（别名: sd21/sd15/zero123/2dgs/threestudio/a/b/c）",
    )
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--full", action="store_true", help="下载 mode 下全部卷（非 only-missing）")
    parser.add_argument(
        "--fix-layout-only",
        action="store_true",
        help="仅整理 staging -> 正式目录，不下载",
    )
    args = parser.parse_args()

    token = get_token()
    login(token)

    if args.shard is not None:
        selected = [resolve_shard(args.shard)]
    else:
        selected = shards_for_mode(args.mode)

    if args.fix_layout_only:
        normalize_layout()
        verify_all(selected)
        return

    if args.only_missing:
        todo = missing_shards(selected)
        if not todo:
            print("所选分卷均已齐全。")
        else:
            for shard in todo:
                download_shard(shard, token)
    elif args.full or args.shard is not None:
        for shard in selected:
            download_shard(shard, token)
    else:
        todo = missing_shards(selected)
        if todo:
            for shard in todo:
                download_shard(shard, token)
        else:
            for shard in selected:
                download_shard(shard, token)

    normalize_layout()
    verify_all(selected)
    print("\n完成。建议: export HW3_ROOT=/mnt/workspace/HW3")


if __name__ == "__main__":
    main()
