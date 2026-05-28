"""Offline evaluation: D zero-shot, B val (ID), ABC val (ID)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch

from .datasets import (
    make_abc_train_val_datasets,
    make_b_train_val_datasets,
    make_d_test_dataset,
    make_dataloader,
)
from .eval_core import EvalAccumulator, EvalProgressLogger, run_eval_loop
from .episode_meta import load_action_groups
from .metrics import evaluate_l1
from .norm_stats import load_action_std_from_checkpoint
from .paths import ensure_lerobot_on_path
from .splits import TEST_ENV, TRAIN_ENVS, load_splits

SplitName = Literal["d_test", "b_val", "abc_val"]


def load_trained_policy(checkpoint_dir: Path, device: torch.device):
    ensure_lerobot_on_path()
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=checkpoint_dir)
    return policy, preprocessor


def _envs_for_split(split: SplitName) -> list[str]:
    if split == "d_test":
        return [TEST_ENV]
    if split == "b_val":
        return ["B"]
    if split == "abc_val":
        return list(TRAIN_ENVS)
    raise ValueError(split)


def _count_episodes(split: SplitName, splits) -> int:
    if split == "d_test":
        return len(splits.test_episodes(TEST_ENV))
    if split == "b_val":
        return len(splits.val_episodes("B"))
    return sum(len(splits.val_episodes(e)) for e in TRAIN_ENVS)


def eval_offline(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    split: SplitName = "d_test",
    batch_size: int = 8,
    num_workers: int = 4,
    max_batches: int | None = None,
    model_tag: str | None = None,
) -> dict:
    """
    Single-pass offline eval. Writes eval_summary.json, eval_episodes.csv,
    eval_per_task.csv, eval.log (progress).
    """
    splits = load_splits()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocessor = load_trained_policy(checkpoint_dir, device)
    action_std = load_action_std_from_checkpoint(checkpoint_dir)

    envs = _envs_for_split(split)
    groups = load_action_groups(envs[0])
    chunk_size = int(policy.config.chunk_size)
    accumulator = EvalAccumulator(
        envs=envs,
        action_std=action_std,
        groups=groups,
        chunk_size=chunk_size,
        top_k_tasks=10,
        top_k_episodes=20,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = EvalProgressLogger(
        output_dir / "eval.log",
        total_episodes=_count_episodes(split, splits),
        flush_every=50,
    )

    total_seconds = 0.0
    if split == "d_test":
        ds = make_d_test_dataset(splits)
        loader = make_dataloader(ds, batch_size, num_workers, shuffle=False)
        total_seconds += run_eval_loop(
            policy,
            preprocessor,
            loader,
            device,
            env=TEST_ENV,
            accumulator=accumulator,
            progress=progress,
            max_batches=max_batches,
        )
    elif split == "b_val":
        _, val_ds = make_b_train_val_datasets(splits)
        loader = make_dataloader(val_ds, batch_size, num_workers, shuffle=False)
        total_seconds += run_eval_loop(
            policy,
            preprocessor,
            loader,
            device,
            env="B",
            accumulator=accumulator,
            progress=progress,
            max_batches=max_batches,
        )
    elif split == "abc_val":
        _, val_ds = make_abc_train_val_datasets(splits)
        # ConcatDataset: evaluate per env with local episode_index.
        for env in TRAIN_ENVS:
            from .datasets import _make_env_dataset, resolve_act_delta_timestamps

            delta = resolve_act_delta_timestamps(env)
            sub = _make_env_dataset(env, splits.val_episodes(env), delta)
            loader = make_dataloader(sub, batch_size, num_workers, shuffle=False)
            total_seconds += run_eval_loop(
                policy,
                preprocessor,
                loader,
                device,
                env=env,
                accumulator=accumulator,
                progress=progress,
                max_batches=max_batches,
            )

    meta = {
        "checkpoint": str(checkpoint_dir),
        "split": split,
        "model_tag": model_tag or checkpoint_dir.parent.parent.name,
        "eval_envs": envs,
        "num_test_episodes": _count_episodes(split, splits),
        "elapsed_seconds": total_seconds,
    }
    summary = accumulator.finalize_summary(meta)
    accumulator.write_outputs(output_dir, summary)

    # Backward-compatible alias for old scripts
    (output_dir / "eval_D_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": summary["checkpoint"],
                "test_env": TEST_ENV if split == "d_test" else None,
                "split": split,
                "num_test_episodes": summary["num_episodes"],
                "l1_loss": summary["l1_norm"],
                "loss": summary["l1_norm"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress.on_progress(summary["num_episodes"], summary["l1_norm"])
    progress._write(f"[{progress._ts()}] eval done l1_norm={summary['l1_norm']:.6f}\n")

    return summary


def eval_zero_shot_on_d(
    checkpoint_dir: Path,
    output_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Backward-compatible wrapper (experiment 3, env D only)."""
    summary = eval_offline(
        checkpoint_dir,
        output_dir,
        split="d_test",
        batch_size=batch_size,
        num_workers=num_workers,
        max_batches=max_batches,
    )
    return {"l1_loss": summary["l1_norm"], "loss": summary["l1_norm"]}
