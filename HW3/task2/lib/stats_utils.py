"""Merge per-environment normalization stats for ABC mixed training (scheme B)."""

from __future__ import annotations

import copy

from .paths import ensure_lerobot_on_path, env_dataset_root
from .splits import TRAIN_ENVS, SplitSpec


def _scale_stats_counts(
    stats: dict[str, dict], train_fraction: float
) -> dict[str, dict]:
    """Down-weight stats counts so aggregate_stats matches train-episode mass only."""
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    scaled = copy.deepcopy(stats)
    for feature_stats in scaled.values():
        if "count" in feature_stats:
            feature_stats["count"] = feature_stats["count"] * train_fraction
    return scaled


def get_merged_abc_train_stats(splits: SplitSpec) -> dict[str, dict]:
    """
    Weighted merge of A/B/C meta.stats using train-episode fractions (scheme B).

    Each environment's per-feature ``count`` is scaled by
    len(train_episodes) / total_episodes before calling LeRobot ``aggregate_stats``.
    """
    ensure_lerobot_on_path()
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    scaled_stats_list: list[dict[str, dict]] = []
    for env in TRAIN_ENVS:
        meta = LeRobotDatasetMetadata(
            repo_id=f"local/calvin_{env}",
            root=env_dataset_root(env),
        )
        if meta.stats is None:
            raise FileNotFoundError(
                f"Missing stats for env {env} at {env_dataset_root(env) / 'meta' / 'stats.json'}"
            )
        n_train = len(splits.train_episodes(env))
        n_total = meta.total_episodes
        if n_train <= 0 or n_total <= 0:
            raise ValueError(f"Invalid episode counts for env {env}: train={n_train}, total={n_total}")
        train_fraction = n_train / n_total
        scaled_stats_list.append(_scale_stats_counts(meta.stats, train_fraction))

    return aggregate_stats(scaled_stats_list)
