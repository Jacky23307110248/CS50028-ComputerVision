import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import csv
from dataloader import load_mnist
from model import mlp_logits, softmax, get_activation
from utils import (
    load_model,
    accuracy,
    plot_confusion,
    per_class_metrics,
    plot_history,
    confusion_matrix,
    save_confusion_csv,
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def build_args():
    parser = argparse.ArgumentParser(description="Evaluate best MLP on Fashion-MNIST test set")
    parser.add_argument("--data_path", type=str, default="./FashionMNIST/raw")
    parser.add_argument("--model_path", type=str, default="./artifacts/best_model_full.npy")
    parser.add_argument("--fallback_model_path", type=str, default="./artifacts/best_model_val.npy")
    parser.add_argument("--history_path", type=str, default="./artifacts/history_full.csv")
    parser.add_argument("--fallback_history_path", type=str, default="./artifacts/history_val.csv")
    parser.add_argument("--fig_dir", type=str, default="./figures")
    parser.add_argument("--error_cases", type=int, default=5)
    return parser.parse_args()


def main():
    args = build_args()
    print("[Notice] Test set should be used only for final evaluation. Avoid tuning hyper-parameters from test results.")
    if args.error_cases < 0:
        raise ValueError(f"--error_cases must be >= 0, got {args.error_cases}")
    os.makedirs(args.fig_dir, exist_ok=True)
    error_dir = os.path.join(args.fig_dir, "errors")
    os.makedirs(error_dir, exist_ok=True)

    _, _, x_test, y_test, _, y_test_cls = load_mnist(data_path=args.data_path)
    selected_model_path = args.model_path
    selected_history_path = args.history_path
    if not os.path.exists(selected_model_path):
        if os.path.exists(args.fallback_model_path):
            print(
                f"[Warning] Primary model not found: {selected_model_path}. "
                f"Fallback to validation model: {args.fallback_model_path}"
            )
            selected_model_path = args.fallback_model_path
            selected_history_path = args.fallback_history_path
        else:
            raise FileNotFoundError(
                f"{selected_model_path} not found, and fallback model "
                f"{args.fallback_model_path} also not found. Please run grid_search.py first."
            )
    model_name = os.path.basename(selected_model_path).lower()
    if "val" in model_name and "full" not in model_name:
        print(
            "[Warning] Current model seems from validation-stage training "
            "(filename contains 'val'). For final submission, prefer full-data retrained weights."
        )
    w1, b1, w2, b2, activation = load_model(selected_model_path)
    activation_fn = get_activation(activation)
    logits_test = mlp_logits(x_test, w1, b1, w2, b2, activation_fn)
    if not np.all(np.isfinite(logits_test)):
        raise FloatingPointError("Non-finite logits detected during test inference.")
    # Use the same numerically stable softmax path as training-side CE.
    y_pred_prob = softmax(logits_test)
    if not np.all(np.isfinite(y_pred_prob)):
        raise FloatingPointError("Non-finite probabilities detected during test inference.")
    test_acc = accuracy(y_pred_prob, y_test)
    print(f"Test Acc: {test_acc:.4f}")

    y_pred_cls = np.argmax(y_pred_prob, axis=1)
    cm = confusion_matrix(y_test_cls, y_pred_cls, num_classes=10)
    print("Confusion Matrix (count):")
    print(cm)
    save_confusion_csv(cm, path=os.path.join(args.fig_dir, "confusion_matrix.csv"))
    print(f"Confusion CSV saved to: {os.path.join(args.fig_dir, 'confusion_matrix.csv')}")
    plot_confusion(
        y_test_cls,
        y_pred_cls,
        normalize=False,
        path=os.path.join(args.fig_dir, "confusion_count.png"),
        class_names=CLASS_NAMES,
    )
    plot_confusion(
        y_test_cls,
        y_pred_cls,
        normalize=True,
        path=os.path.join(args.fig_dir, "confusion_norm.png"),
        class_names=CLASS_NAMES,
    )

    metrics = per_class_metrics(y_test_cls, y_pred_cls)
    for i, (p, r, f1) in enumerate(metrics):
        print(f"[{i}] {CLASS_NAMES[i]:>12s} | Precision {p:.4f} | Recall {r:.4f} | F1 {f1:.4f}")

    num_show = min(32, w1.shape[1])
    w1_norm = np.linalg.norm(w1, axis=0)
    top_idx = np.argsort(-w1_norm)[:num_show]
    w_show = w1[:, top_idx].T
    w_show = (w_show - w_show.min()) / (w_show.max() - w_show.min() + 1e-12)
    rows, cols = 4, 8
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    for idx, ax in enumerate(axes.ravel()):
        if idx < num_show:
            ax.imshow(w_show[idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"Neuron {top_idx[idx]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "weight1_grid.png"))
    plt.close()

    err_idx = np.where(y_pred_cls != y_test_cls)[0]
    err_conf = y_pred_prob[err_idx, y_pred_cls[err_idx]]
    sorted_err = err_idx[np.argsort(-err_conf)]
    error_csv_path = os.path.join(args.fig_dir, "error_cases.csv")
    with open(error_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_id", "true_name", "pred_id", "pred_name", "confidence"])
        for i in sorted_err[: args.error_cases]:
            writer.writerow(
                [
                    int(i),
                    int(y_test_cls[i]),
                    CLASS_NAMES[int(y_test_cls[i])],
                    int(y_pred_cls[i]),
                    CLASS_NAMES[int(y_pred_cls[i])],
                    float(y_pred_prob[i, y_pred_cls[i]]),
                ]
            )

    for i in sorted_err[: args.error_cases]:
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        conf = y_pred_prob[i, y_pred_cls[i]]
        plt.title(
            f"True {CLASS_NAMES[int(y_test_cls[i])]} | Pred {CLASS_NAMES[int(y_pred_cls[i])]} | Conf {conf:.3f}"
        )
        plt.savefig(
            os.path.join(
                error_dir, f"err_{i}_t{int(y_test_cls[i])}_p{int(y_pred_cls[i])}.png"
            )
        )
        plt.close()

    # Prefer validation-stage history for visualization so history.png
    # contains train/val curves used for model selection.
    history_for_plot = selected_history_path
    if os.path.exists(args.fallback_history_path):
        history_for_plot = args.fallback_history_path
    if os.path.exists(history_for_plot):
        plot_history(
            history_path=history_for_plot,
            save_path=os.path.join(args.fig_dir, "history.png"),
        )


if __name__ == "__main__":
    main()