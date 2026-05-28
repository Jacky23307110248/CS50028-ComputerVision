"""Metrics logging: SwanLab + local JSONL (no test-set metrics during training)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class MetricsLogger:
    def __init__(self, run_name: str, log_dir: Path, use_swanlab: bool = True):
        self.run_name = run_name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.use_swanlab = use_swanlab
        self._swan = None
        if use_swanlab:
            try:
                import swanlab

                self._swan = swanlab
                self._swan.init(
                    project="CVHW3",
                    workspace="23307110248JackyH",
                    experiment_name=run_name,
                    logdir=str(log_dir),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] SwanLab init failed ({exc}); logging to JSONL only.")
                self.use_swanlab = False

    def log(self, step: int, metrics: dict[str, float | int]) -> None:
        row: dict[str, Any] = {"step": step, **metrics}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if self.use_swanlab and self._swan is not None:
            self._swan.log(metrics, step=step)

    def finish(self) -> None:
        if self.use_swanlab and self._swan is not None:
            self._swan.finish()


def _predict_actions(policy, batch: dict):
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    actions_hat, _ = policy.model(batch)
    return batch[ACTION], actions_hat


def _act_batch_errors(
    policy,
    batch: dict,
    *,
    action_std: list[float] | None = None,
    groups: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> dict:
    """
    Per-batch errors in normalized action space.

    Returns sums/counts for frame-weighted aggregation (not means).
    """
    from lerobot.utils.constants import ACTION

    actions, actions_hat = _predict_actions(policy, batch)
    abs_err = (actions - actions_hat).abs()
    valid_mask = (~batch["action_is_pad"]).unsqueeze(-1)
    masked = abs_err * valid_mask

    valid_elements = valid_mask.sum() * abs_err.shape[-1]
    sum_l1 = float(masked.sum())
    count = int(valid_elements.clamp_min(1).item())

    per_dim_sum = masked.sum(dim=(0, 1)).detach().cpu().tolist()

    out: dict = {
        "sum_l1": sum_l1,
        "count": count,
        "per_dim_sum": per_dim_sum,
    }

    if action_std is not None:
        std = torch.tensor(action_std, device=abs_err.device, dtype=abs_err.dtype)
        denorm = masked * std
        out["sum_mae_denorm"] = float(denorm.sum())
        out["per_dim_denorm_sum"] = denorm.sum(dim=(0, 1)).detach().cpu().tolist()

    if groups is not None:
        pos, rot, grip = groups
        out["sum_pos"] = float(masked[..., pos[0] : pos[1]].sum())
        out["sum_rot"] = float(masked[..., rot[0] : rot[1]].sum())
        out["sum_grip"] = float(masked[..., grip[0] : grip[1]].sum())
        pos_n = int(valid_mask.sum() * (pos[1] - pos[0]))
        rot_n = int(valid_mask.sum() * (rot[1] - rot[0]))
        grip_n = int(valid_mask.sum() * (grip[1] - grip[0]))
        out["count_pos"] = max(pos_n, 1)
        out["count_rot"] = max(rot_n, 1)
        out["count_grip"] = max(grip_n, 1)

    return out


def _baseline_batch_errors(
    batch: dict,
    *,
    mode: str,
    action_std: list[float] | None = None,
    groups: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> dict:
    """Zero / mean baselines in normalized space (mean ≈ 0 under MEAN_STD)."""
    from lerobot.utils.constants import ACTION

    actions = batch[ACTION]
    if mode == "zero":
        actions_hat = torch.zeros_like(actions)
    elif mode == "mean":
        actions_hat = torch.zeros_like(actions)
    else:
        raise ValueError(mode)

    abs_err = (actions - actions_hat).abs()
    valid_mask = (~batch["action_is_pad"]).unsqueeze(-1)
    masked = abs_err * valid_mask
    valid_elements = valid_mask.sum() * abs_err.shape[-1]
    out: dict = {
        "sum_l1": float(masked.sum()),
        "count": int(valid_elements.clamp_min(1).item()),
        "per_dim_sum": masked.sum(dim=(0, 1)).detach().cpu().tolist(),
    }
    if action_std is not None:
        std = torch.tensor(action_std, device=abs_err.device, dtype=abs_err.dtype)
        denorm = masked * std
        out["sum_mae_denorm"] = float(denorm.sum())
        out["per_dim_denorm_sum"] = denorm.sum(dim=(0, 1)).detach().cpu().tolist()
    if groups is not None:
        pos, rot, grip = groups
        out["sum_pos"] = float(masked[..., pos[0] : pos[1]].sum())
        out["sum_rot"] = float(masked[..., rot[0] : rot[1]].sum())
        out["sum_grip"] = float(masked[..., grip[0] : grip[1]].sum())
        out["count_pos"] = max(int(valid_mask.sum() * (pos[1] - pos[0])), 1)
        out["count_rot"] = max(int(valid_mask.sum() * (rot[1] - rot[0])), 1)
        out["count_grip"] = max(int(valid_mask.sum() * (grip[1] - grip[0])), 1)
    return out


# Action-chunk horizons for Phase 3 (ACT chunk_size=100).
EVAL_HORIZONS = (1, 5, 10, 20, 50, 100)


def _chunk_errors_from_abs_err(
    abs_err: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict:
    """
    Chunk-step and horizon sums from (B, T, D) abs errors and (B, T, 1) mask.

    Returns lists aligned with timesteps / EVAL_HORIZONS for frame-weighted means.
    """
    masked = abs_err * valid_mask
    _, chunk_size, action_dim = abs_err.shape

    step_sum: list[float] = []
    step_count: list[int] = []
    for t in range(chunk_size):
        step_mask = valid_mask[:, t, :]
        n = int(step_mask.sum() * action_dim)
        step_sum.append(float(masked[:, t, :].sum()))
        step_count.append(n)

    horizon_sum: list[float] = []
    horizon_count: list[int] = []
    for k in EVAL_HORIZONS:
        kk = min(k, chunk_size)
        h_mask = valid_mask[:, :kk, :]
        n = int(h_mask.sum() * action_dim)
        horizon_sum.append(float(masked[:, :kk, :].sum()))
        horizon_count.append(max(n, 0))

    mid = chunk_size // 2
    first_mask = valid_mask[:, :mid, :]
    second_mask = valid_mask[:, mid:, :]
    first_n = int(first_mask.sum() * action_dim)
    second_n = int(second_mask.sum() * action_dim)

    return {
        "chunk_step_sum": step_sum,
        "chunk_step_count": step_count,
        "horizon_k": list(EVAL_HORIZONS),
        "horizon_sum": horizon_sum,
        "horizon_count": horizon_count,
        "first_half_sum": float(masked[:, :mid, :].sum()),
        "first_half_count": max(first_n, 0),
        "second_half_sum": float(masked[:, mid:, :].sum()),
        "second_half_count": max(second_n, 0),
    }


def _act_batch_l1(policy, batch: dict) -> float:
    """Action L1 in eval mode. ACTPolicy.forward() cannot be used here when use_vae=True
    because the VAE encoder only runs in training mode (mu/log_sigma are None in eval)."""
    err = _act_batch_errors(policy, batch)
    return err["sum_l1"] / err["count"]


@torch.no_grad()
def evaluate_l1(
    policy,
    preprocessor,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Offline L1 on a dataloader (val only — never pass test D during training)."""
    policy.eval()
    total_l1 = 0.0
    n = 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _prepare_batch(batch, preprocessor, device)
        l1 = _act_batch_l1(policy, batch)
        total_l1 += l1
        n += 1
    policy.train()
    if n == 0:
        return {"l1_loss": 0.0, "loss": 0.0}
    avg_l1 = total_l1 / n
    return {"l1_loss": avg_l1, "loss": avg_l1}


def _prepare_batch(batch, preprocessor, device: torch.device):
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.dtype == torch.uint8:
            batch[key] = val.to(dtype=torch.float32) / 255.0
    batch = preprocessor(batch)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch
