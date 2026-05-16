import argparse
from copy import deepcopy

from defaults import DEFAULTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Task3 segmentation runner")
    parser.add_argument("--mode", choices=["train", "check_data"], default=DEFAULTS["mode"])
    parser.add_argument("--data-root", type=str, default=DEFAULTS["data_root"])
    parser.add_argument("--output-root", type=str, default=DEFAULTS["output_root"])
    parser.add_argument("--run-name", type=str, default=DEFAULTS["run_name"])

    parser.add_argument("--img-size", type=int, default=DEFAULTS["img_size"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=DEFAULTS["scheduler"])
    parser.add_argument("--cosine-t-max", type=int, default=DEFAULTS["cosine_t_max"])
    parser.add_argument("--cosine-eta-min", type=float, default=DEFAULTS["cosine_eta_min"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--device", type=str, default=DEFAULTS["device"])

    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], default=DEFAULTS["loss"])
    parser.add_argument("--ce-weight", type=float, default=DEFAULTS["ce_weight"])
    parser.add_argument("--dice-weight", type=float, default=DEFAULTS["dice_weight"])
    parser.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    return parser


def parse_config() -> dict:
    parser = build_parser()
    args = parser.parse_args()
    cfg = deepcopy(DEFAULTS)
    cfg.update(vars(args))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["img_size"] <= 0:
        raise ValueError("img_size must be > 0")
    if cfg["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg["epochs"] <= 0 and cfg["mode"] == "train":
        raise ValueError("epochs must be > 0 in train mode")
    if cfg["patience"] < 0:
        raise ValueError("patience must be >= 0")
    if cfg["lr"] <= 0:
        raise ValueError("lr must be > 0")
    if cfg["cosine_t_max"] < 0:
        raise ValueError("cosine_t_max must be >= 0")
    if cfg["cosine_eta_min"] < 0:
        raise ValueError("cosine_eta_min must be >= 0")
    if cfg["ce_weight"] < 0 or cfg["dice_weight"] < 0:
        raise ValueError("ce_weight and dice_weight must be >= 0")
