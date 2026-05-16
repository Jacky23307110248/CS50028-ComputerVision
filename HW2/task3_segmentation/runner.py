from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config_schema import parse_config
from core.dataset import OxfordPetSegmentation, build_dataloaders
from core.engine import train_one_epoch, validate
from core.losses import CombinedLoss
from core.utils import (
    append_metrics_csv,
    init_logger,
    init_metrics_csv,
    pick_device,
    prepare_run_dir,
    save_config,
    save_result,
    set_seed,
)
from models import build_model

'''
def save_results_plot(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_miou = [r["train_miou"] for r in rows]
    val_miou = [r["val_miou"] for r in rows]

    fig, (ax_loss, ax_miou) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)

    ax_loss.plot(epochs, train_loss, color="#E53935", linewidth=2, label="train")
    ax_loss.plot(epochs, val_loss, color="#C62828", linewidth=2, linestyle="--", label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss (train / val)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best")

    ax_miou.plot(epochs, train_miou, color="#E53935", linewidth=2, label="train")
    ax_miou.plot(epochs, val_miou, color="#C62828", linewidth=2, linestyle="--", label="val")
    ax_miou.set_xlabel("epoch")
    ax_miou.set_ylabel("mIoU")
    ax_miou.set_title("mIoU (train / val)")
    ax_miou.set_ylim(0.0, 1.0)
    ax_miou.grid(True, alpha=0.3)
    ax_miou.legend(loc="best")

    fig.savefig(run_dir / "results.png", dpi=160)
    plt.close(fig)
'''

def run_check_data(cfg: dict):
    print("[check_data] building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"[check_data] train samples: {len(train_loader.dataset)}")
    print(f"[check_data] val samples: {len(val_loader.dataset)}")

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    train_images, train_masks = train_batch
    val_images, val_masks = val_batch

    print(
        "[check_data] train batch shapes:",
        tuple(train_images.shape),
        tuple(train_masks.shape),
    )
    print(
        "[check_data] val batch shapes:",
        tuple(val_images.shape),
        tuple(val_masks.shape),
    )
    print("[check_data] dataloader works.")


def run_train(cfg: dict):
    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])

    run_dir = prepare_run_dir(cfg)
    logger = init_logger(run_dir)
    save_config(run_dir, cfg)
    init_metrics_csv(run_dir)

    logger.info("Run dir: %s", str(run_dir))
    logger.info("Device: %s", device)
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info("Train samples: %d | Val samples: %d", len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(name="unet", num_classes=3).to(device)
    criterion = CombinedLoss(
        loss_type=cfg["loss"], ce_weight=cfg["ce_weight"], dice_weight=cfg["dice_weight"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = None
    if cfg["scheduler"] == "cosine":
        t_max = cfg["cosine_t_max"] if cfg["cosine_t_max"] > 0 else cfg["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=cfg["cosine_eta_min"]
        )
    logger.info(
        "Optimizer: AdamW | Scheduler: %s",
        "CosineAnnealingLR" if scheduler is not None else "None",
    )

    best_miou = -1.0
    best_epoch = -1
    best_path = run_dir / "best.pt"
    no_improve_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        append_metrics_csv(run_dir, epoch, train_stats, val_stats)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_miou": train_stats["miou"],
                "val_loss": val_stats["loss"],
                "val_miou": val_stats["miou"],
            }
        )

        logger.info(
            "Epoch %d/%d | lr=%.6g | train_loss=%.4f train_mIoU=%.4f | val_loss=%.4f val_mIoU=%.4f",
            epoch,
            cfg["epochs"],
            current_lr,
            train_stats["loss"],
            train_stats["miou"],
            val_stats["loss"],
            val_stats["miou"],
        )

        if val_stats["miou"] > best_miou:
            best_miou = val_stats["miou"]
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve_epochs += 1

        if cfg["patience"] > 0 and no_improve_epochs >= cfg["patience"]:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d, best_epoch=%d, best_val_mIoU=%.4f)",
                epoch,
                cfg["patience"],
                best_epoch,
                best_miou,
            )
            break

        if cfg["save_every"] > 0 and (epoch % cfg["save_every"] == 0):
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch}.pt")

    test_ds = OxfordPetSegmentation(cfg["data_root"], split="test", img_size=cfg["img_size"], augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_stats = validate(model, test_loader, criterion, device)

    result = {
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_ckpt": str(best_path),
        "test_loss": test_stats["loss"],
        "test_mIoU": test_stats["miou"],
    }
    save_result(run_dir, result)
    #save_results_plot(run_dir, history)
    logger.info(
        "Training finished. Best epoch=%d, best val mIoU=%.4f | test_loss=%.4f test_mIoU=%.4f",
        best_epoch,
        best_miou,
        test_stats["loss"],
        test_stats["miou"],
    )


def main():
    cfg = parse_config()
    Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)

    if cfg["mode"] == "check_data":
        run_check_data(cfg)
        return
    run_train(cfg)


if __name__ == "__main__":
    main()
