import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import json
from datetime import datetime

def he_init(in_dim, out_dim, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    w = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * np.sqrt(2.0 / in_dim)
    return w.astype(np.float32)


def xavier_init(in_dim, out_dim, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    w = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * np.sqrt(1.0 / in_dim)
    return w.astype(np.float32)

# clip the gradient of l2 norm
def clip_grad(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * max_norm / norm
    return grad

# clip the gradient of global L2 norm
def clip_grads_global(grads, max_norm=1.0):
    total_sq = 0.0
    for g in grads:
        total_sq += float(np.sum(g * g))
    global_norm = np.sqrt(total_sq)
    if global_norm <= max_norm or global_norm == 0.0:
        return grads
    scale = max_norm / global_norm
    return [g * scale for g in grads]

# cosine backfire learning rate
def cosine_lr(epoch, total_epoch, lr_max=0.1):
    denom = max(total_epoch - 1, 1)
    return lr_max * 0.5 * (1 + np.cos(np.pi * epoch / denom))

def save_model(w1, b1, w2, b2, activation="relu", path="best_model.npy"):
    model_dir = os.path.dirname(path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    np.save(path, {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "activation": activation})

def load_model(path="best_model.npy"):
    d = np.load(path, allow_pickle=True).item()
    if "activation" not in d:
        raise KeyError(
            f"Missing key 'activation' in model file: {path}. "
            "Please re-save weights with activation metadata."
        )
    activation = d["activation"]
    return d["W1"], d["b1"], d["W2"], d["b2"], activation


def save_history(history, path="history.csv"):
    if not history:
        return
    history_dir = os.path.dirname(path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)
    keys = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_best_meta(meta, path="best_meta.json"):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    payload = dict(meta)
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_split_indices(train_idx, val_idx, path="split_indices.npz"):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.savez(path, train_idx=train_idx, val_idx=val_idx)


def plot_history(history_path="history.csv", save_path="history.png"):
    data = np.genfromtxt(history_path, delimiter=",", names=True)
    if data.size == 0:
        return
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    epochs = data["epoch"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    train_loss_key = "train_total_loss" if "train_total_loss" in data.dtype.names else "train_loss"
    val_loss_key = "val_total_loss" if "val_total_loss" in data.dtype.names else "val_loss"
    plt.plot(epochs, data[train_loss_key], label=train_loss_key)
    plt.plot(epochs, data[val_loss_key], label=val_loss_key)
    if "train_ce_loss" in data.dtype.names:
        plt.plot(epochs, data["train_ce_loss"], "--", label="train_ce_loss")
    if "val_ce_loss" in data.dtype.names:
        plt.plot(epochs, data["val_ce_loss"], "--", label="val_ce_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    if "val_acc" in data.dtype.names:
        plt.plot(epochs, data["val_acc"], label="val_acc")
    plt.plot(epochs, data["train_acc"], label="train_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_data_split(x, y, val_ratio=0.2, seed=42, return_indices=False):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = np.random.default_rng(seed)
    y_cls = np.argmax(y, axis=1)
    num_classes = y.shape[1]
    tr_parts = []
    val_parts = []
    for c in range(num_classes):
        cls_idx = np.where(y_cls == c)[0]
        cls_idx = rng.permutation(cls_idx)
        split = int((1.0 - val_ratio) * len(cls_idx))
        tr_parts.append(cls_idx[:split])
        val_parts.append(cls_idx[split:])
    tr_idx = np.concatenate(tr_parts)
    val_idx = np.concatenate(val_parts)
    tr_idx = rng.permutation(tr_idx)
    val_idx = rng.permutation(val_idx)
    if return_indices:
        return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx], tr_idx, val_idx
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(y_true, y_pred, normalize=False, path="confusion.png", class_names=None):
    cm = confusion_matrix(y_true, y_pred, num_classes=10)
    mat = cm.astype(np.float64)
    fmt = "g"
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, np.maximum(row_sum, 1), where=row_sum > 0)
        fmt = ".2f"
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto",
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confusion_csv(cm, path="confusion_matrix.csv"):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.savetxt(path, cm, delimiter=",", fmt="%d")


def per_class_metrics(y_true, y_pred, num_classes=10):
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    metrics = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics.append((precision, recall, f1))
    return metrics