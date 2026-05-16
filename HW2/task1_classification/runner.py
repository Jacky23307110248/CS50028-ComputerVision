import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch

from dataset import build_dataloaders, build_test_loader
from engine import train_one_epoch, validate_one_epoch
from losses import build_classification_loss
from model_factory import build_model, build_optimizer, build_scheduler
from plot_utils import save_curves_plot
from utils import load_yaml, prepare_experiment_dir, save_json, save_yaml, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Task1 classification runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--override-exp-name", type=str, default=None)
    parser.add_argument("--override-seed", type=int, default=None)
    return parser.parse_args()


def train_and_eval(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    test_loader = build_test_loader(cfg)
    model = build_model(cfg).to(device)
    criterion = build_classification_loss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    exp_dir = prepare_experiment_dir(cfg["output_root"], cfg["exp_name"])
    save_yaml(cfg, os.path.join(exp_dir, "config_used.yaml"))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(exp_dir, "best.pth")
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "config": cfg,
                },
                best_ckpt_path,
            )

        print(
            f"Epoch [{epoch + 1}/{cfg['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    elapsed = time.time() - start_time
    curves_path = os.path.join(exp_dir, "curves.npz")
    np.savez(curves_path, **{k: np.array(v) for k, v in history.items()})
    save_curves_plot(curves_path, os.path.join(exp_dir, "figs"))

    # Final test evaluation: load best checkpoint and evaluate once on official test split.
    ckpt = torch.load(best_ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    test_loss, test_acc = validate_one_epoch(model, test_loader, criterion, device)

    metrics = {
        "exp_name": cfg["exp_name"],
        "seed": cfg["seed"],
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
        "elapsed_sec": float(elapsed),
        "device": str(device),
        "test_eval_status": "ok",
        "test_eval_updated_at": datetime.now().isoformat(timespec="seconds"),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_device": str(device),
    }
    save_json(metrics, os.path.join(exp_dir, "metrics.json"))
    return {"exp_dir": exp_dir, **metrics}


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # 默认优先使用项目下的 Oxford-IIIT 目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_candidates = [
        os.path.join(base_dir, "..", "Oxford-IIIT"),
        os.path.join(base_dir, "dataset"),
    ]
    default_data_root = None
    for candidate in default_candidates:
        candidate_abs = os.path.abspath(candidate)
        if os.path.isdir(candidate_abs):
            default_data_root = candidate_abs
            break
    if default_data_root is None:
        default_data_root = os.path.abspath(default_candidates[0])

    cfg.setdefault("data_root", default_data_root)
    cfg.setdefault("output_root", os.path.join(base_dir, "outputs"))
    cfg.setdefault("val_ratio", 0.1)
    cfg.setdefault("split_seed", cfg.get("seed", 42))
    if not os.path.isabs(cfg["data_root"]):
        cfg["data_root"] = os.path.join(base_dir, cfg["data_root"])
    if not os.path.isabs(cfg["output_root"]):
        cfg["output_root"] = os.path.join(base_dir, cfg["output_root"])

    # 实验调度脚本可通过环境变量覆盖超参数
    if os.environ.get("T1_BACKBONE_LR") is not None:
        cfg["backbone_lr"] = float(os.environ["T1_BACKBONE_LR"])
    if os.environ.get("T1_HEAD_LR") is not None:
        cfg["head_lr"] = float(os.environ["T1_HEAD_LR"])
    if os.environ.get("T1_EPOCHS") is not None:
        cfg["epochs"] = int(os.environ["T1_EPOCHS"])
    if os.environ.get("T1_PRETRAINED") is not None:
        cfg["pretrained"] = os.environ["T1_PRETRAINED"] == "1"
    if os.environ.get("T1_ATTENTION") is not None:
        cfg["attention"] = os.environ["T1_ATTENTION"]

    if args.override_exp_name is not None:
        cfg["exp_name"] = args.override_exp_name
    if args.override_seed is not None:
        cfg["seed"] = args.override_seed

    result = train_and_eval(cfg)
    print("Experiment done:", result["exp_dir"])


if __name__ == "__main__":
    main()

