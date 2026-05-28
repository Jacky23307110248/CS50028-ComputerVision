"""Episode / task metadata (LeRobot v2.1 jsonl and v3.0 parquet)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .paths import dataset_codebase_version, ensure_lerobot_on_path, env_dataset_root

# CALVIN layout (same across A/B/C/D); v3.0 drops meta/modality.json after convert.
CALVIN_ACTION_GROUPS = {
    "pos": (0, 3),   # eef_pos_delta
    "rot": (3, 6),   # eef_rot_delta
    "grip": (6, 7),  # gripper_close
}


@dataclass(frozen=True)
class ActionGroups:
    pos: tuple[int, int] = CALVIN_ACTION_GROUPS["pos"]
    rot: tuple[int, int] = CALVIN_ACTION_GROUPS["rot"]
    grip: tuple[int, int] = CALVIN_ACTION_GROUPS["grip"]


def load_action_groups(env: str) -> ActionGroups:  # noqa: ARG001
    """CALVIN 7-d action layout is fixed; env kept for API compatibility."""
    return ActionGroups()


def _row_tasks_to_strings(tasks_raw: Any, task_index_to_name: dict[int, str] | None) -> list[str]:
    if tasks_raw is None:
        return []
    if hasattr(tasks_raw, "tolist"):
        tasks_raw = tasks_raw.tolist()
    if isinstance(tasks_raw, str):
        return [tasks_raw]
    if not isinstance(tasks_raw, (list, tuple)):
        return [str(tasks_raw)]

    out: list[str] = []
    for t in tasks_raw:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, (int, float)) and task_index_to_name is not None:
            out.append(task_index_to_name.get(int(t), f"task_{int(t)}"))
        else:
            out.append(str(t))
    return out


def _load_episode_table_v30(env: str) -> dict[int, dict]:
    ensure_lerobot_on_path()
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    root = env_dataset_root(env)
    meta = LeRobotDatasetMetadata(repo_id=f"local/calvin_{env}", root=root)
    task_index_to_name: dict[int, str] = {}
    if meta.tasks is not None and len(meta.tasks) > 0:
        for task_name, row in meta.tasks.iterrows():
            task_index_to_name[int(row["task_index"])] = str(task_name)

    table: dict[int, dict] = {}
    for i in range(meta.total_episodes):
        row = meta.episodes[i]
        ep_idx = int(row["episode_index"])
        tasks = _row_tasks_to_strings(row.get("tasks"), task_index_to_name)
        length = row.get("length")
        table[ep_idx] = {
            "episode_index": ep_idx,
            "tasks": tasks,
            "length": int(length) if length is not None else None,
        }
    return table


def _load_episode_table_v21(env: str) -> dict[int, dict]:
    path = env_dataset_root(env) / "meta" / "episodes.jsonl"
    table: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        table[int(row["episode_index"])] = row
    return table


def load_episode_table(env: str) -> dict[int, dict]:
    """Episode index -> {tasks, length, ...} for per-episode / per-task eval stats."""
    version = dataset_codebase_version(env)
    if version == "v3.0":
        return _load_episode_table_v30(env)
    return _load_episode_table_v21(env)


def episode_task_name(episode_row: dict) -> str:
    tasks = episode_row.get("tasks") or []
    if not tasks:
        return "unknown"
    return str(tasks[0])
