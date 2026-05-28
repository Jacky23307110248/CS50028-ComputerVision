"""LeRobot dataset builders with explicit episode lists (no test leakage)."""

from __future__ import annotations

from torch.utils.data import ConcatDataset, DataLoader

from .paths import ensure_lerobot_on_path, env_dataset_root
from .splits import SplitSpec, TRAIN_ENVS


def _make_env_dataset(env: str, episodes: list[int], delta_timestamps, image_transforms=None):
    ensure_lerobot_on_path()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=f"local/calvin_{env}",
        root=env_dataset_root(env),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        return_uint8=True,
        # pyav on ROCm: avoid torchcodec/FFmpeg issues; slightly looser tolerance for keyframe seek.
        video_backend="pyav",
        tolerance_s=1e-3,
    )


def resolve_act_delta_timestamps(env: str = "B"):
    ensure_lerobot_on_path()
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.act.configuration_act import ACTConfig

    meta = LeRobotDatasetMetadata(repo_id=f"local/calvin_{env}", root=env_dataset_root(env))
    return resolve_delta_timestamps(ACTConfig(), meta)


def make_b_train_val_datasets(splits: SplitSpec):
    delta = resolve_act_delta_timestamps("B")
    train_ds = _make_env_dataset("B", splits.train_episodes("B"), delta)
    val_ds = _make_env_dataset("B", splits.val_episodes("B"), delta)
    return train_ds, val_ds


def make_abc_train_val_datasets(splits: SplitSpec):
    """Train/val on A,B,C train/val episodes only. D is never loaded here."""
    train_parts = []
    val_parts = []
    for env in TRAIN_ENVS:
        delta = resolve_act_delta_timestamps(env)
        train_parts.append(_make_env_dataset(env, splits.train_episodes(env), delta))
        val_parts.append(_make_env_dataset(env, splits.val_episodes(env), delta))
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def make_d_test_dataset(splits: SplitSpec):
    """Test set: environment D only. For eval script, not training."""
    delta = resolve_act_delta_timestamps("D")
    return _make_env_dataset("D", splits.test_episodes("D"), delta)


def make_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ensure_lerobot_on_path()
    from lerobot.utils.collate import lerobot_collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lerobot_collate_fn,
    )
