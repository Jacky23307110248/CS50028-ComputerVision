import argparse
import os
import numpy as np
from dataloader import load_mnist
from model import Tape, mlp_logits, stable_softmax_ce, l2_loss, get_activation
from utils import (
    he_init,
    xavier_init,
    clip_grads_global,
    cosine_lr,
    save_model,
    accuracy,
    save_history,
    get_data_split,
    save_best_meta,
    save_split_indices,
)

INPUT_DIM = 784
OUT_DIM = 10
DEFAULT_SEED = 42


def _assert_finite(name, arr, epoch=None, step=None):
    if np.all(np.isfinite(arr)):
        return
    loc = []
    if epoch is not None:
        loc.append(f"epoch={epoch}")
    if step is not None:
        loc.append(f"step={step}")
    loc_str = ", ".join(loc)
    if loc_str:
        loc_str = f" ({loc_str})"
    raise FloatingPointError(f"Non-finite detected in {name}{loc_str}.")


def run_train_with_params(
    lr_max,
    hidden_dim,
    batch_size,
    l2_lambda,
    epochs=30,
    patience=7,
    activation="relu",
    seed=DEFAULT_SEED,
    model_path="best_model.npy",
    history_path="history.csv",
    save_best=True,
    data_path="./FashionMNIST/raw",
    use_validation_split=True,
    val_ratio=0.2,
    min_delta=1e-4,
    grad_clip_norm=1.0,
    best_meta_path=None,
    config_path=None,
    split_path=None,
    save_final_if_no_val=True,
    preloaded_train_data=None,
    split_seed=None,
):
    if lr_max <= 0:
        raise ValueError(f"lr_max must be > 0, got {lr_max}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if l2_lambda < 0:
        raise ValueError(f"l2_lambda must be >= 0, got {l2_lambda}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if patience <= 0:
        raise ValueError(f"patience must be > 0, got {patience}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if min_delta < 0:
        raise ValueError(f"min_delta must be >= 0, got {min_delta}")
    if grad_clip_norm <= 0:
        raise ValueError(f"grad_clip_norm must be > 0, got {grad_clip_norm}")
    if split_seed is not None and split_seed < 0:
        raise ValueError(f"split_seed must be >= 0 when provided, got {split_seed}")
    if save_best and not use_validation_split and not save_final_if_no_val:
        raise ValueError(
            "save_best=True with no validation requires save_final_if_no_val=True "
            "to save final-epoch weights explicitly."
        )

    rng = np.random.default_rng(seed)
    effective_split_seed = seed if split_seed is None else split_seed
    activation_fn = get_activation(activation)
    if config_path:
        save_best_meta(
            {
                "stage": "train",
                "data_path": data_path,
                "lr_max": lr_max,
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "l2_lambda": l2_lambda,
                "epochs": epochs,
                "patience": patience,
                "activation": activation,
                "seed": seed,
                "model_path": model_path,
                "history_path": history_path,
                "best_meta_path": best_meta_path,
                "split_path": split_path,
                "use_validation_split": use_validation_split,
                "val_ratio": val_ratio,
                "min_delta": min_delta,
                "grad_clip_norm": grad_clip_norm,
                "save_best": save_best,
                "save_final_if_no_val": save_final_if_no_val,
                "split_seed": split_seed,
                "effective_split_seed": effective_split_seed,
            },
            path=config_path,
        )

    if preloaded_train_data is not None:
        if (
            not isinstance(preloaded_train_data, (tuple, list))
            or len(preloaded_train_data) != 2
        ):
            raise ValueError(
                "preloaded_train_data must be a tuple/list of (x_train, y_train)."
            )
        x_train, y_train = preloaded_train_data
    else:
        x_train, y_train, _, _, _, _ = load_mnist(data_path=data_path)
    if use_validation_split:
        x_tr, y_tr, x_val, y_val, tr_idx, val_idx = get_data_split(
            x_train,
            y_train,
            val_ratio=val_ratio,
            seed=effective_split_seed,
            return_indices=True,
        )
        if split_path:
            save_split_indices(tr_idx, val_idx, path=split_path)
    else:
        x_tr, y_tr = x_train, y_train
        x_val, y_val = None, None

    if activation == "relu":
        w1 = he_init(INPUT_DIM, hidden_dim, rng=rng)
        w2 = he_init(hidden_dim, OUT_DIM, rng=rng)
        init_name = "he"
    else:
        w1 = xavier_init(INPUT_DIM, hidden_dim, rng=rng)
        w2 = xavier_init(hidden_dim, OUT_DIM, rng=rng)
        init_name = "xavier"
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    b2 = np.zeros(OUT_DIM, dtype=np.float32)

    best_score = -np.inf
    best_total_loss = np.inf
    counter = 0
    history = []

    for epoch in range(epochs):
        lr = cosine_lr(epoch, epochs, lr_max)
        idx = rng.permutation(len(x_tr))
        x_shuffle, y_shuffle = x_tr[idx], y_tr[idx]
        epoch_loss = 0.0
        steps = 0

        for i in range(0, len(x_shuffle), batch_size):
            x_batch = x_shuffle[i:i + batch_size]
            y_batch = y_shuffle[i:i + batch_size]

            tape = Tape()
            logits = mlp_logits(x_batch, w1, b1, w2, b2, activation_fn, tape=tape)
            ce_loss, _ = stable_softmax_ce(logits, y_batch, tape=tape)
            reg_loss = l2_loss(w1, w2, l2_lambda)
            total_loss = ce_loss + reg_loss
            _assert_finite("ce_loss", np.array(ce_loss), epoch=epoch, step=steps)
            _assert_finite("reg_loss", np.array(reg_loss), epoch=epoch, step=steps)
            _assert_finite("total_loss", np.array(total_loss), epoch=epoch, step=steps)

            tape.backward(ce_loss, grad_out=1.0)

            dw1 = tape.get_grad(w1) + l2_lambda * w1
            db1 = tape.get_grad(b1)
            dw2 = tape.get_grad(w2) + l2_lambda * w2
            db2 = tape.get_grad(b2)
            _assert_finite("dw1", dw1, epoch=epoch, step=steps)
            _assert_finite("db1", db1, epoch=epoch, step=steps)
            _assert_finite("dw2", dw2, epoch=epoch, step=steps)
            _assert_finite("db2", db2, epoch=epoch, step=steps)
            dw1, db1, dw2, db2 = clip_grads_global(
                [dw1, db1, dw2, db2], max_norm=grad_clip_norm
            )

            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2
            _assert_finite("w1", w1, epoch=epoch, step=steps)
            _assert_finite("b1", b1, epoch=epoch, step=steps)
            _assert_finite("w2", w2, epoch=epoch, step=steps)
            _assert_finite("b2", b2, epoch=epoch, step=steps)

            epoch_loss += float(total_loss)
            steps += 1

        train_logits = mlp_logits(x_tr, w1, b1, w2, b2, activation_fn)
        train_ce, train_prob = stable_softmax_ce(train_logits, y_tr)
        if use_validation_split:
            val_logits = mlp_logits(x_val, w1, b1, w2, b2, activation_fn)
            val_ce, val_prob = stable_softmax_ce(val_logits, y_val)
            val_acc = float(accuracy(val_prob, y_val))
            val_total = float(val_ce + l2_loss(w1, w2, l2_lambda))
            tracked_acc = val_acc
            tracked_total = val_total
            tracked_name = "val"
        else:
            val_ce = float("nan")
            val_prob = None
            val_acc = float("nan")
            val_total = float("nan")
            tracked_acc = float(accuracy(train_prob, y_tr))
            tracked_total = float(train_ce + l2_loss(w1, w2, l2_lambda))
            tracked_name = "train"
        train_total = float(train_ce + l2_loss(w1, w2, l2_lambda))
        train_acc = float(accuracy(train_prob, y_tr))

        history.append({
            "epoch": epoch,
            "lr": float(lr),
            "train_ce_loss": float(train_ce),
            "val_ce_loss": float(val_ce),
            "train_total_loss": train_total,
            "val_total_loss": val_total,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "tracked_total_loss": tracked_total,
            "tracked_acc": tracked_acc,
            "tracked_metric_source": tracked_name,
            "batch_loss": epoch_loss / max(steps, 1),
            "weight_norm_w1": float(np.linalg.norm(w1)),
            "weight_norm_w2": float(np.linalg.norm(w2)),
        })

        print(
            f"Epoch {epoch:02d} | LR {lr:.5f} | "
            f"Train Loss {train_total:.4f} Acc {train_acc:.4f} | "
            f"Tracked({tracked_name}) Loss {tracked_total:.4f} Acc {tracked_acc:.4f} | "
            f"Init {init_name} | Seed {seed}"
        )

        score = tracked_acc
        improved = False
        if score > best_score + min_delta:
            improved = True
        elif abs(score - best_score) <= min_delta and tracked_total < best_total_loss - 1e-12:
            improved = True

        if improved:
            best_score = score
            best_total_loss = tracked_total
            counter = 0
            if save_best and use_validation_split:
                save_model(w1, b1, w2, b2, activation=activation, path=model_path)
                if best_meta_path:
                    save_best_meta(
                        {
                            "best_epoch": epoch,
                            "best_val_acc": val_acc,
                            "best_val_total_loss": val_total,
                            "model_path": model_path,
                            "history_path": history_path,
                            "activation": activation,
                            "seed": seed,
                            "hidden_dim": hidden_dim,
                            "batch_size": batch_size,
                            "lr_max": lr_max,
                            "l2_lambda": l2_lambda,
                            "val_ratio": val_ratio,
                            "use_validation_split": use_validation_split,
                            "split_seed": split_seed,
                            "effective_split_seed": effective_split_seed,
                            "selection_rule": "maximize val_acc with min_delta, tie-break by lower val_total_loss",
                        },
                        path=best_meta_path,
                    )
        else:
            counter += 1
            if use_validation_split and counter >= patience:
                print("Early stop!")
                break

    if save_best and not use_validation_split and save_final_if_no_val:
        save_model(w1, b1, w2, b2, activation=activation, path=model_path)
        if best_meta_path:
            save_best_meta(
                {
                    "best_epoch": history[-1]["epoch"] if history else -1,
                    "best_tracked_acc": history[-1]["tracked_acc"] if history else 0.0,
                    "best_tracked_total_loss": history[-1]["tracked_total_loss"] if history else 0.0,
                    "tracked_metric_source": "train",
                    "model_path": model_path,
                    "history_path": history_path,
                    "activation": activation,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "batch_size": batch_size,
                    "lr_max": lr_max,
                    "l2_lambda": l2_lambda,
                    "val_ratio": 0.0,
                    "use_validation_split": False,
                    "split_seed": split_seed,
                    "effective_split_seed": effective_split_seed,
                    "selection_note": "Saved final-epoch weights from full-data training (no validation split).",
                },
                path=best_meta_path,
            )

    save_history(history, history_path)
    return best_score, history


def build_args():
    parser = argparse.ArgumentParser(description="Train 3-layer MLP for Fashion-MNIST")
    parser.add_argument("--data_path", type=str, default="./FashionMNIST/raw")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model_path", type=str, default="./artifacts/best_model_val.npy")
    parser.add_argument("--history_path", type=str, default="./artifacts/history_val.csv")
    parser.add_argument("--best_meta_path", type=str, default="./artifacts/best_meta_val.json")
    parser.add_argument("--config_path", type=str, default="./artifacts/run_config.json")
    parser.add_argument("--split_path", type=str, default="./artifacts/split_indices.npz")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_validation_split", dest="use_validation_split", action="store_true")
    parser.add_argument("--no_validation_split", dest="use_validation_split", action="store_false")
    parser.set_defaults(use_validation_split=True)
    return parser.parse_args()


def main():
    args = build_args()
    model_dir = os.path.dirname(args.model_path)
    history_dir = os.path.dirname(args.history_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    best_acc, _ = run_train_with_params(
        lr_max=args.lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        l2_lambda=args.l2,
        epochs=args.epochs,
        patience=args.patience,
        activation=args.activation,
        seed=args.seed,
        model_path=args.model_path,
        history_path=args.history_path,
        save_best=True,
        data_path=args.data_path,
        use_validation_split=args.use_validation_split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        min_delta=args.min_delta,
        grad_clip_norm=args.grad_clip_norm,
        best_meta_path=args.best_meta_path,
        config_path=args.config_path,
        split_path=args.split_path,
    )
    metric_name = "validation accuracy" if args.use_validation_split else "tracked training accuracy"
    print(f"Best {metric_name}: {best_acc:.4f}")


if __name__ == "__main__":
    main()