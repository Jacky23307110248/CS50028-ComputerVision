"""验证 ModelScope Token 与 sSzHox/task1_Data 数据集是否可用。

本地 / DSW 均可运行。不要把 Token 写进代码或提交 git。

CMD:
  set MODELSCOPE_API_TOKEN=ms-xxxx
  python verify_task1_modelscope.py

DSW:
  export MODELSCOPE_API_TOKEN=ms-xxxx
  export MODELSCOPE_ACCESS_TOKEN=$MODELSCOPE_API_TOKEN
  python verify_task1_modelscope.py
"""

from __future__ import annotations

import os
import sys

REPO_ID = "sSzHox/task1_Data"


def get_token() -> str:
    token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get(
        "MODELSCOPE_ACCESS_TOKEN"
    )
    if not token:
        print("请先设置 MODELSCOPE_API_TOKEN")
        sys.exit(1)
    return token


def main() -> None:
    token = get_token()

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    print("[1/2] HubApi.login: OK")

    from modelscope.hub.snapshot_download import snapshot_download

    path = snapshot_download(REPO_ID, repo_type="dataset", token=token)
    print(f"[2/2] snapshot_download {REPO_ID}: OK")
    print(f"      缓存路径: {path}")
    print("\n说明:")
    print("  - 若数据集刚创建、尚未上传，这里只会看到 README/.gitattributes，属正常。")
    print("  - 题目一 bulk 资源请用 upload_task1_to_workspace.py，不要用 MsDataset.load。")
    print("  - MsDataset.load 需额外依赖 addict，且适合结构化表格数据，不适合本作业大文件分卷。")


if __name__ == "__main__":
    main()
