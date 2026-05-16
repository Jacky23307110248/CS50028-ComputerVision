import os

import matplotlib.pyplot as plt
import numpy as np


def save_curves_plot(curves_npz_path: str, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    data = np.load(curves_npz_path)
    epochs = np.arange(1, len(data["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data["train_loss"], label="train_loss")
    plt.plot(epochs, data["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, data["train_acc"], label="train_acc")
    plt.plot(epochs, data["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "curves.png"), dpi=200)
    plt.close()

