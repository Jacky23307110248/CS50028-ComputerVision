from pathlib import Path


DEFAULTS = {
    "seed": 42,
    "data_root": str((Path(__file__).resolve().parent / "../Oxford-IIIT").resolve()),
    "output_root": str((Path(__file__).resolve().parent / "outputs").resolve()),
    "run_name": "",
    "img_size": 256,
    "batch_size": 8,
    "num_workers": 2,
    "epochs": 50,
    "patience": 10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "cosine_t_max": 0,
    "cosine_eta_min": 1e-6,
    "device": "cuda",
    "loss": "ce",
    "ce_weight": 1.0,
    "dice_weight": 1.0,
    "save_every": 0,
    "mode": "train",
}
