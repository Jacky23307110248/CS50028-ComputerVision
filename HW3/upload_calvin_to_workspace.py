# upload_calvin_to_workspace.py
# 在本地 Windows 运行：上传到魔搭数据集 sSzHox/hw3-calvin-data
# 上传完成后，在 PAI-DSW 开发机用下方「下载脚本」拉到 /mnt/workspace/HW3/data/

import os
import sys
from pathlib import Path

from modelscope.hub.api import HubApi

# ========== 配置 ==========
REPO_ID = "sSzHox/hw3-calvin-data"
LOCAL_DIR = Path(r"D:\大三下\计算机视觉\HW3\data\calvin_task_ABC_D")
PATH_IN_REPO = "calvin_task_ABC_D"

SHARDS = [
    "calvin_task_ABC_D_lerobot_3_4",
]

IGNORE = ["**/.git/**", "__pycache__/**", "*.pyc", ".DS_Store"]
MAX_WORKERS = 2
# ==========================


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


def main() -> None:
    if not LOCAL_DIR.is_dir():
        raise FileNotFoundError(f"本地目录不存在: {LOCAL_DIR}")

    missing = [s for s in SHARDS if not (LOCAL_DIR / s).is_dir()]
    if missing:
        raise FileNotFoundError(f"缺少分卷目录: {missing}")

    git_dir = LOCAL_DIR / ".git"
    if git_dir.exists():
        print(f"跳过本地 git 缓存: {git_dir}（无需上传，避免 5 万文件/目录限制）")

    token = get_token()
    api = HubApi()
    api.login(token)

    print(f"目标数据集: {REPO_ID}")
    print(f"仓库内根路径: {PATH_IN_REPO}/")
    print(f"分 {len(SHARDS)} 卷依次上传，请耐心等待…\n")

    for i, shard in enumerate(SHARDS, 1):
        shard_path = LOCAL_DIR / shard
        target = f"{PATH_IN_REPO}/{shard}"
        print(f"[{i}/{len(SHARDS)}] 上传 {shard} -> {target}/")

        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=str(shard_path),
            path_in_repo=target,
            repo_type="dataset",
            ignore_patterns=IGNORE,
            max_workers=MAX_WORKERS,
            commit_message=f"Upload CALVIN shard {shard} for HW3",
        )
        print(f"[{i}/{len(SHARDS)}] 完成: {shard}\n")

    attrs = LOCAL_DIR / ".gitattributes"
    if attrs.is_file():
        print("上传 .gitattributes …")
        api.upload_file(
            repo_id=REPO_ID,
            path_or_fileobj=str(attrs),
            path_in_repo=f"{PATH_IN_REPO}/.gitattributes",
            repo_type="dataset",
            commit_message="Upload .gitattributes for HW3 CALVIN data",
        )

    print("全部上传完成。")
    print("请到 PAI-DSW 开发机 Terminal 执行下方下载脚本。")


if __name__ == "__main__":
    main()

# ----- 开发机下载（Terminal 粘贴运行）-----
#
# import os
# from modelscope.hub.snapshot_download import snapshot_download
#
# os.makedirs("/mnt/workspace/HW3/data", exist_ok=True)
# snapshot_download(
#     "sSzHox/hw3-calvin-data",
#     repo_type="dataset",
#     local_dir="/mnt/workspace/HW3/data",
# )
# print(os.listdir("/mnt/workspace/HW3/data/calvin_task_ABC_D"))
#
# export HW3_ROOT=/mnt/workspace/HW3
# python task2/scripts/check_setup.py
