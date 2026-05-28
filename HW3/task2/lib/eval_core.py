"""Offline evaluation accumulator (single pass, many metrics)."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .episode_meta import ActionGroups, episode_task_name, load_action_groups, load_episode_table
from .metrics import (
    EVAL_HORIZONS,
    _act_batch_errors,
    _baseline_batch_errors,
    _chunk_errors_from_abs_err,
    _predict_actions,
    _prepare_batch,
)


@dataclass
class RunningSums:
    sum_l1: float = 0.0
    count: int = 0
    sum_mae_denorm: float = 0.0
    per_dim_sum: list[float] = field(default_factory=lambda: [0.0] * 7)
    per_dim_denorm_sum: list[float] = field(default_factory=lambda: [0.0] * 7)
    sum_pos: float = 0.0
    sum_rot: float = 0.0
    sum_grip: float = 0.0
    count_pos: int = 0
    count_rot: int = 0
    count_grip: int = 0

    def update(self, d: dict) -> None:
        self.sum_l1 += d["sum_l1"]
        self.count += d["count"]
        if "sum_mae_denorm" in d:
            self.sum_mae_denorm += d["sum_mae_denorm"]
        for i, v in enumerate(d["per_dim_sum"]):
            self.per_dim_sum[i] += v
        if "per_dim_denorm_sum" in d:
            for i, v in enumerate(d["per_dim_denorm_sum"]):
                self.per_dim_denorm_sum[i] += v
        if "sum_pos" in d:
            self.sum_pos += d["sum_pos"]
            self.sum_rot += d["sum_rot"]
            self.sum_grip += d["sum_grip"]
            self.count_pos += d["count_pos"]
            self.count_rot += d["count_rot"]
            self.count_grip += d["count_grip"]

    def mean_l1(self) -> float:
        return self.sum_l1 / max(self.count, 1)

    def mean_mae_denorm(self) -> float:
        return self.sum_mae_denorm / max(self.count, 1)


@dataclass
class EpisodeAgg:
    sum_l1: float = 0.0
    count: int = 0
    env: str = ""
    episode_index: int = 0

    def mean(self) -> float:
        return self.sum_l1 / max(self.count, 1)


class EvalProgressLogger:
    def __init__(self, log_path: Path, total_episodes: int, flush_every: int = 50):
        self.log_path = log_path
        self.total_episodes = total_episodes
        self.flush_every = flush_every
        self._last_logged = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write(f"[{self._ts()}] eval start total_episodes={total_episodes}\n")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _write(self, msg: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(msg)

    def on_progress(self, episodes_done: int, running_l1: float, last_key: str | None = None) -> None:
        if episodes_done <= self._last_logged:
            return
        if episodes_done % self.flush_every != 0 and episodes_done != self.total_episodes:
            return
        self._last_logged = episodes_done
        extra = f" last_episode={last_key}" if last_key else ""
        self._write(
            f"[{self._ts()}] episodes_done={episodes_done}/{self.total_episodes} "
            f"running_l1={running_l1:.6f}{extra}\n"
        )


class ChunkRunningSums:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.step_sum = [0.0] * chunk_size
        self.step_count = [0] * chunk_size
        self.horizon_sum = [0.0] * len(EVAL_HORIZONS)
        self.horizon_count = [0] * len(EVAL_HORIZONS)
        self.first_half_sum = 0.0
        self.first_half_count = 0
        self.second_half_sum = 0.0
        self.second_half_count = 0

    def update(self, d: dict) -> None:
        for t, s in enumerate(d["chunk_step_sum"]):
            if t < self.chunk_size:
                self.step_sum[t] += s
                self.step_count[t] += d["chunk_step_count"][t]
        for i, s in enumerate(d["horizon_sum"]):
            self.horizon_sum[i] += s
            self.horizon_count[i] += d["horizon_count"][i]
        self.first_half_sum += d["first_half_sum"]
        self.first_half_count += d["first_half_count"]
        self.second_half_sum += d["second_half_sum"]
        self.second_half_count += d["second_half_count"]

    def mean_step_l1(self, t: int) -> float:
        return self.step_sum[t] / max(self.step_count[t], 1)

    def mean_horizon_l1(self, i: int) -> float:
        return self.horizon_sum[i] / max(self.horizon_count[i], 1)


class EvalAccumulator:
    def __init__(
        self,
        *,
        envs: list[str],
        action_std: list[float],
        groups: ActionGroups,
        chunk_size: int = 100,
        top_k_tasks: int = 10,
        top_k_episodes: int = 20,
    ):
        self.envs = envs
        self.action_std = action_std
        self.groups = groups
        self.chunk_size = chunk_size
        self.top_k_tasks = top_k_tasks
        self.top_k_episodes = top_k_episodes
        self.group_slices = (groups.pos, groups.rot, groups.grip)

        self.policy = RunningSums()
        self.zero = RunningSums()
        self.mean = RunningSums()
        self.chunk = ChunkRunningSums(chunk_size)
        self.episodes: dict[str, EpisodeAgg] = {}
        self.task_sums: dict[str, RunningSums] = defaultdict(RunningSums)
        self._episode_tables = {e: load_episode_table(e) for e in envs}

    def _episode_key(self, env: str, episode_index: int) -> str:
        return f"{env}:{episode_index}"

    def _groups_tuple(self):
        return self.group_slices

    @torch.no_grad()
    def consume_batch(
        self,
        policy,
        batch: dict,
        *,
        env: str,
        mode: str = "policy",
    ) -> str | None:
        if mode == "policy":
            err = _act_batch_errors(
                policy,
                batch,
                action_std=self.action_std,
                groups=self._groups_tuple(),
            )
            target = self.policy
        elif mode in ("zero", "mean"):
            err = _baseline_batch_errors(
                batch,
                mode=mode,
                action_std=self.action_std,
                groups=self._groups_tuple(),
            )
            target = self.zero if mode == "zero" else self.mean
        else:
            raise ValueError(mode)

        target.update(err)

        if mode != "policy":
            return

        actions, actions_hat = _predict_actions(policy, batch)
        valid_mask = (~batch["action_is_pad"]).unsqueeze(-1)
        abs_err = (actions - actions_hat).abs()
        self.chunk.update(_chunk_errors_from_abs_err(abs_err, valid_mask))
        abs_err_masked = abs_err * valid_mask
        ep_idx = batch["episode_index"].reshape(-1).detach().cpu().tolist()
        last_key: str | None = None

        for b, epi in enumerate(ep_idx):
            epi = int(epi)
            key = self._episode_key(env, epi)
            sample_mask = valid_mask[b]
            n = int(sample_mask.sum() * abs_err.shape[-1])
            if n == 0:
                continue
            s = float(abs_err_masked[b].sum())
            if key not in self.episodes:
                self.episodes[key] = EpisodeAgg(env=env, episode_index=epi)
            self.episodes[key].sum_l1 += s
            self.episodes[key].count += n
            last_key = key

            task = episode_task_name(self._episode_tables[env][epi])
            self.task_sums[task].sum_l1 += s
            self.task_sums[task].count += n

        return last_key

    def finalize_summary(self, meta: dict[str, Any]) -> dict[str, Any]:
        ep_means = [e.mean() for e in self.episodes.values()]
        ep_means_sorted = sorted(ep_means)
        n_ep = len(ep_means_sorted)

        def pct(p: float) -> float:
            if n_ep == 0:
                return 0.0
            idx = min(int(round(p * (n_ep - 1))), n_ep - 1)
            return ep_means_sorted[idx]

        worst = sorted(self.episodes.values(), key=lambda e: e.mean(), reverse=True)
        best = sorted(self.episodes.values(), key=lambda e: e.mean())

        l1 = self.policy.mean_l1()
        b0 = self.zero.mean_l1()
        bm = self.mean.mean_l1()

        summary: dict[str, Any] = {
            **meta,
            "aggregation": "frame_weighted",
            "denorm_source": "checkpoint_action_std",
            "baseline_note": "zero/mean baselines in normalized space (MEAN_STD => mean ~ 0)",
            "chunk_size": self.chunk_size,
            "eval_protocol": "open_loop_offline",
            "l1_norm": l1,
            "loss": l1,
            "mae_denorm": self.policy.mean_mae_denorm(),
            "l1_pos": self.policy.sum_pos / max(self.policy.count_pos, 1),
            "l1_rot": self.policy.sum_rot / max(self.policy.count_rot, 1),
            "l1_grip": self.policy.sum_grip / max(self.policy.count_grip, 1),
            "baseline_zero_l1": b0,
            "baseline_mean_l1": bm,
            "margin_vs_zero_pct": (b0 - l1) / b0 * 100.0 if b0 > 0 else 0.0,
            "margin_vs_mean_pct": (bm - l1) / bm * 100.0 if bm > 0 else 0.0,
            "l1_chunk_first_half": self.chunk.first_half_sum / max(self.chunk.first_half_count, 1),
            "l1_chunk_second_half": self.chunk.second_half_sum / max(self.chunk.second_half_count, 1),
            "num_episodes": n_ep,
            "num_frames": self.policy.count,
            "l1_ep_mean": sum(ep_means) / max(n_ep, 1),
            "l1_ep_median": pct(0.5),
            "l1_ep_p90": pct(0.9),
            "l1_ep_p95": pct(0.95),
            "per_dim_l1": [v / max(self.policy.count, 1) for v in self.policy.per_dim_sum],
            "per_dim_mae_denorm": [
                v / max(self.policy.count, 1) for v in self.policy.per_dim_denorm_sum
            ],
            "worst_episode_ids": [
                {"env": e.env, "episode_index": e.episode_index, "l1": e.mean()}
                for e in worst[: self.top_k_episodes]
            ],
            "best_episode_ids": [
                {"env": e.env, "episode_index": e.episode_index, "l1": e.mean()}
                for e in best[: self.top_k_episodes]
            ],
        }
        return summary

    def write_outputs(self, output_dir: Path, summary: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        with (output_dir / "eval_episodes.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["env", "episode_index", "task", "l1_norm", "num_frames"],
            )
            w.writeheader()
            for key in sorted(self.episodes.keys()):
                e = self.episodes[key]
                task = episode_task_name(self._episode_tables[e.env][e.episode_index])
                w.writerow(
                    {
                        "env": e.env,
                        "episode_index": e.episode_index,
                        "task": task,
                        "l1_norm": e.mean(),
                        "num_frames": e.count,
                    }
                )

        task_rows = []
        for task, sums in self.task_sums.items():
            task_rows.append((task, sums.mean_l1(), sums.count))
        task_rows.sort(key=lambda x: x[1], reverse=True)

        with (output_dir / "eval_per_task.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["rank", "task", "l1_norm", "num_frames", "hard_or_easy"],
            )
            w.writeheader()
            for i, (task, l1, n) in enumerate(task_rows[: self.top_k_tasks], start=1):
                w.writerow(
                    {
                        "rank": i,
                        "task": task,
                        "l1_norm": l1,
                        "num_frames": n,
                        "hard_or_easy": "hard",
                    }
                )
            easy_start = max(len(task_rows) - self.top_k_tasks, 0)
            for j, (task, l1, n) in enumerate(task_rows[easy_start:], start=1):
                w.writerow(
                    {
                        "rank": j,
                        "task": task,
                        "l1_norm": l1,
                        "num_frames": n,
                        "hard_or_easy": "easy",
                    }
                )

        with (output_dir / "eval_chunk_steps.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["chunk_step", "l1_norm", "num_action_elements"]
            )
            w.writeheader()
            for t in range(self.chunk_size):
                w.writerow(
                    {
                        "chunk_step": t,
                        "l1_norm": self.chunk.mean_step_l1(t),
                        "num_action_elements": self.chunk.step_count[t],
                    }
                )

        with (output_dir / "eval_horizons.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["horizon_k", "l1_norm", "num_action_elements"])
            w.writeheader()
            for i, k in enumerate(EVAL_HORIZONS):
                w.writerow(
                    {
                        "horizon_k": k,
                        "l1_norm": self.chunk.mean_horizon_l1(i),
                        "num_action_elements": self.chunk.horizon_count[i],
                    }
                )


def run_eval_loop(
    policy,
    preprocessor,
    dataloader,
    device: torch.device,
    *,
    env: str,
    accumulator: EvalAccumulator,
    progress: EvalProgressLogger | None = None,
    max_batches: int | None = None,
) -> None:
    policy.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _prepare_batch(batch, preprocessor, device)
            last_key = accumulator.consume_batch(policy, batch, env=env, mode="policy")
            accumulator.consume_batch(policy, batch, env=env, mode="zero")
            accumulator.consume_batch(policy, batch, env=env, mode="mean")
            if progress is not None:
                progress.on_progress(
                    len(accumulator.episodes),
                    accumulator.policy.mean_l1(),
                    last_key=last_key,
                )
    policy.train()
    elapsed = time.perf_counter() - t0
    return elapsed
