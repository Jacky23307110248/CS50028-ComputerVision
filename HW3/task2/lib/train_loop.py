"""ACT training loop with strict train/val only (test env D excluded)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .datasets import make_dataloader
from .metrics import MetricsLogger, evaluate_l1
from .paths import OUTPUTS_DIR, ensure_lerobot_on_path
from .splits import load_splits
from .stats_utils import get_merged_abc_train_stats


@dataclass
class TrainHyperparams:
    """Matches lerobot TrainPipelineConfig defaults (batch=8, steps=100k)."""

    batch_size: int = 8
    steps: int = 100_000
    log_freq: int = 200
    eval_freq: int = 20_000
    save_freq: int = 20_000
    num_workers: int = 4
    seed: int = 1000
    grad_clip_norm: float = 10.0
    lr: float = 1e-5  # ACT preset default


def _set_seed(seed: int) -> None:
    ensure_lerobot_on_path()
    from lerobot.utils.random_utils import set_seed as lr_set_seed

    lr_set_seed(seed)


class _DatasetMetaView:
    """Expose merged stats while delegating other fields to an underlying LeRobot meta."""

    def __init__(self, base_meta, stats: dict):
        self._base_meta = base_meta
        self.stats = stats

    def __getattr__(self, name: str):
        return getattr(self._base_meta, name)


def build_act_policy(train_dataset: Dataset):
    ensure_lerobot_on_path()
    from lerobot.configs import FeatureType
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features

    # B-only (and any single LeRobotDataset): unchanged path — no ABC stats merge.
    if isinstance(train_dataset, ConcatDataset):
        base_meta = train_dataset.datasets[0].meta
        dataset_stats = get_merged_abc_train_stats(load_splits())
        meta = _DatasetMetaView(base_meta, dataset_stats)
    else:
        meta = train_dataset.meta
        dataset_stats = meta.stats

    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    policy_cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = make_policy(cfg=policy_cfg, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, dataset_stats=dataset_stats
    )
    return policy, preprocessor, postprocessor, policy_cfg


def _build_optimizer(policy, steps: int):
    ensure_lerobot_on_path()
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.optim.factory import make_optimizer_and_scheduler

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/calvin"),
        policy=policy.config,
        steps=steps,
        use_policy_training_preset=True,
    )
    cfg.optimizer = policy.config.get_optimizer_preset()
    cfg.scheduler = policy.config.get_scheduler_preset()
    return make_optimizer_and_scheduler(cfg, policy)


def _save_checkpoint(out_dir: Path, step: int, policy, preprocessor, postprocessor, hparams: TrainHyperparams):
    ckpt_dir = out_dir / f"checkpoints/{step:07d}"
    pretrained_dir = ckpt_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained_dir)
    policy.config.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    (pretrained_dir / "training_step.json").write_text(
        json.dumps({"step": step, **asdict(hparams)}), encoding="utf-8"
    )
    (out_dir / "last_checkpoint.txt").write_text(str(pretrained_dir), encoding="utf-8")


def train_act(
    run_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    hparams: TrainHyperparams | None = None,
    use_swanlab: bool = True,
) -> Path:
    """
    Train ACT on train_dataset; validate on val_dataset only.
    Caller must NOT pass environment D data here.
    """
    hparams = hparams or TrainHyperparams()
    _set_seed(hparams.seed)

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hyperparams.json").write_text(
        json.dumps(asdict(hparams), indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocessor, postprocessor, _ = build_act_policy(train_dataset)
    policy.train()
    policy.to(device)

    optimizer, lr_scheduler = _build_optimizer(policy, hparams.steps)

    train_loader = make_dataloader(
        train_dataset, hparams.batch_size, hparams.num_workers, shuffle=True
    )
    val_loader = make_dataloader(
        val_dataset, hparams.batch_size, hparams.num_workers, shuffle=False
    )

    logger = MetricsLogger(run_name, out_dir / "logs", use_swanlab=use_swanlab)
    train_iter = iter(train_loader)
    train_loss_avg = 0.0
    train_l1_avg = 0.0
    log_count = 0

    for step in range(1, hparams.steps + 1):
        t0 = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and val.dtype == torch.uint8:
                batch[key] = val.to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        loss, out = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), hparams.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        train_loss_avg += float(loss.item())
        train_l1_avg += float(out.get("l1_loss", loss.item()))
        log_count += 1
        update_s = time.perf_counter() - t0

        if step % hparams.log_freq == 0:
            metrics = {
                "train/loss": train_loss_avg / log_count,
                "train/l1_loss": train_l1_avg / log_count,
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/update_s": update_s,
            }
            if out.get("kld_loss") is not None:
                metrics["train/kld_loss"] = float(out["kld_loss"])
            logger.log(step, metrics)
            train_loss_avg = train_l1_avg = 0.0
            log_count = 0

        if step % hparams.eval_freq == 0:
            val_metrics = evaluate_l1(policy, preprocessor, val_loader, device)
            logger.log(
                step,
                {f"val/{k}": v for k, v in val_metrics.items()},
            )

        if step % hparams.save_freq == 0 or step == hparams.steps:
            _save_checkpoint(out_dir, step, policy, preprocessor, postprocessor, hparams)

    logger.finish()
    return out_dir
