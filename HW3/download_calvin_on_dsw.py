"""在 PAI-DSW 开发机运行: python download_calvin_on_dsw.py

常见原因：
1) 数据集是「非公开」：必须用 sSzHox 账号的 Token，且传给 snapshot_download(token=...)
2) 第一次 download 时 Hub 还没有 3_4，之后全量重跑会 0 downloaded（只用旧缓存）
3) Hub 上四卷与 calvin_task_ABC_D 的嵌套关系不一致时，需 normalize_layout()
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

REPO_ID = "sSzHox/hw3-calvin-data"
LOCAL_DIR = Path("/mnt/workspace/HW3/data")
ROOT = LOCAL_DIR / "calvin_task_ABC_D"

SHARDS = [
    "calvin_task_ABC_D_lerobot_0_4",
    "calvin_task_ABC_D_lerobot_1_4",
    "calvin_task_ABC_D_lerobot_2_4",
    "calvin_task_ABC_D_lerobot_3_4",
]


def get_token() -> str:
    token = os.environ.get("MODELSCOPE_ACCESS_TOKEN") or os.environ.get(
        "MODELSCOPE_API_TOKEN"
    )
    if not token:
        raise SystemExit(
            "未设置 Token。非公开数据集必须：\n"
            "  export MODELSCOPE_API_TOKEN=ms-xxxx   # 魔搭个人中心 Access Token\n"
            "  export MODELSCOPE_ACCESS_TOKEN=$MODELSCOPE_API_TOKEN"
        )
    return token


def login(token: str) -> None:
    HubApi().login(token)
    print("ModelScope 已登录（私有数据集下载必需）")


def missing_shards() -> list[str]:
    if not ROOT.is_dir():
        return list(SHARDS)
    return [s for s in SHARDS if not (ROOT / s).is_dir()]


def normalize_layout() -> None:
    """若分卷落在 data/<shard> 而非 data/calvin_task_ABC_D/<shard>，挪进去。"""
    ROOT.mkdir(parents=True, exist_ok=True)
    for shard in SHARDS:
        flat = LOCAL_DIR / shard
        nested = ROOT / shard
        if flat.is_dir() and not nested.exists():
            print(f"整理路径: {flat} -> {nested}")
            shutil.move(str(flat), str(nested))


def download_shard(shard: str, token: str) -> None:
    nested = ROOT / shard
    if nested.exists():
        print(f"删除后重下: {nested}")
        shutil.rmtree(nested)

    flat = LOCAL_DIR / shard
    if flat.exists():
        shutil.rmtree(flat)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    patterns = [
        f"calvin_task_ABC_D/{shard}/**",
        f"{shard}/**",
    ]
    for pattern in patterns:
        print(f"尝试拉取: {pattern}")
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            local_dir=str(LOCAL_DIR),
            allow_patterns=[pattern],
            token=token,
        )
        normalize_layout()
        if (ROOT / shard).is_dir():
            print(f"  已成功: {ROOT / shard}")
            return

    print(f"  警告: {shard} 仍未出现在 {ROOT}，检查 Token 权限或网页路径。")


def download_all(token: str) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"snapshot_download 全量 -> {LOCAL_DIR}")
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
        token=token,
    )
    normalize_layout()


def verify_lerobot_layout() -> None:
    normalize_layout()
    if not ROOT.is_dir():
        raise SystemExit(f"缺少目录: {ROOT}\n请先 ls {LOCAL_DIR}")

    print("data/ 下:", sorted(p.name for p in LOCAL_DIR.iterdir()))
    print("calvin_task_ABC_D/ 下:", sorted(p.name for p in ROOT.iterdir()))

    missing = missing_shards()
    if missing:
        raise SystemExit(
            f"仍缺少分卷: {missing}\n"
            "若网页已有 3_4 仍失败，多为私有库 Token：\n"
            "  export MODELSCOPE_API_TOKEN=你的ms-token\n"
            "  python download_calvin_on_dsw.py --shard D"
        )

    for env, shard in zip("ABCD", SHARDS, strict=True):
        info = ROOT / shard / "meta" / "info.json"
        if not info.is_file():
            raise SystemExit(f"缺少: {info}")
        print(f"  OK env {env}: {shard}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--shard", choices=["0", "1", "2", "3", "3_4", "D"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--fix-layout-only",
        action="store_true",
        help="只把 data/<shard> 挪到 data/calvin_task_ABC_D/<shard>，不下载",
    )
    args = parser.parse_args()

    token = get_token()
    login(token)

    if args.fix_layout_only:
        normalize_layout()
        verify_lerobot_layout()
        return

    if args.shard is not None:
        idx = {"0": 0, "1": 1, "2": 2, "3": 3, "3_4": 3, "D": 3}[args.shard]
        download_shard(SHARDS[idx], token)
    elif args.full:
        download_all(token)
    elif args.only_missing:
        miss = missing_shards()
        if not miss:
            print("本地四卷齐全。")
        else:
            for shard in miss:
                download_shard(shard, token)
    else:
        miss = missing_shards()
        if miss:
            for shard in miss:
                download_shard(shard, token)
        else:
            download_all(token)

    verify_lerobot_layout()
    print("\n完成。下一步: python task2/scripts/check_setup.py")


if __name__ == "__main__":
    main()
