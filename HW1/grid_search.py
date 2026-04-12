import csv
import argparse
import os
import time
import json
import numpy as np
from train import run_train_with_params
from dataloader import load_mnist


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_args():
    parser = argparse.ArgumentParser(description="Grid search for MLP hyper-parameters")
    parser.add_argument("--data_path", type=str, default="./FashionMNIST/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_list", type=str, default="42,52,62,72,82")
    parser.add_argument("--search_epochs", type=int, default=20)
    parser.add_argument("--search_patience", type=int, default=5)
    parser.add_argument("--final_epochs", type=int, default=50)
    parser.add_argument("--final_patience", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="./artifacts")
    parser.add_argument("--log_path", type=str, default="./artifacts/search_log.csv")
    parser.add_argument("--best_model_path", type=str, default="./artifacts/best_model_val.npy")
    parser.add_argument("--best_history_path", type=str, default="./artifacts/history_val.csv")
    parser.add_argument("--best_meta_path", type=str, default="./artifacts/best_meta_val.json")
    parser.add_argument("--best_config_path", type=str, default="./artifacts/run_config_val.json")
    parser.add_argument("--best_model_full_path", type=str, default="./artifacts/best_model_full.npy")
    parser.add_argument("--best_history_full_path", type=str, default="./artifacts/history_full.csv")
    parser.add_argument("--best_meta_full_path", type=str, default="./artifacts/best_meta_full.json")
    parser.add_argument("--best_config_full_path", type=str, default="./artifacts/run_config_full.json")
    parser.add_argument("--split_path", type=str, default="./artifacts/split_indices.npz")
    parser.add_argument("--summary_path", type=str, default="./artifacts/search_summary.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_list", type=str, default="0.005,0.01,0.05")
    parser.add_argument("--hidden_list", type=str, default="128,256")
    parser.add_argument("--batch_list", type=str, default="64,128")
    parser.add_argument("--l2_list", type=str, default="1e-4,1e-3")
    parser.add_argument("--act_list", type=str, default="relu,sigmoid")
    parser.add_argument("--retrain_full_data", dest="retrain_full_data", action="store_true")
    parser.add_argument("--no_retrain_full_data", dest="retrain_full_data", action="store_false")
    parser.set_defaults(retrain_full_data=True)
    return parser.parse_args()


def main():
    args = build_args()
    if args.search_epochs <= 0 or args.final_epochs <= 0:
        raise ValueError("search_epochs and final_epochs must be > 0")
    if args.search_patience <= 0 or args.final_patience <= 0:
        raise ValueError("search_patience and final_patience must be > 0")
    if args.topk <= 0:
        raise ValueError("topk must be > 0")
    if args.split_seed < 0:
        raise ValueError("split_seed must be >= 0")
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    lr_list = parse_float_list(args.lr_list)
    hidden_list = parse_int_list(args.hidden_list)
    batch_list = parse_int_list(args.batch_list)
    l2_list = parse_float_list(args.l2_list)
    act_list = [x.strip() for x in args.act_list.split(",") if x.strip()]
    if args.seed_list.strip():
        seed_list = parse_int_list(args.seed_list)
    else:
        seed_list = [args.seed]
    if not seed_list:
        raise ValueError("seed_list must contain at least one seed")
    x_train, y_train, _, _, _, _ = load_mnist(data_path=args.data_path)

    best_cfg = None
    best_acc = -1.0
    all_trials = []

    with open(args.log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lr",
                "hidden",
                "batch",
                "l2",
                "activation",
                "seed_count",
                "val_acc_mean",
                "val_acc_std",
                "best_val_acc_single_seed",
                "best_epoch_at_best_seed",
                "elapsed_sec",
                "seed_details",
            ]
        )
        for lr in lr_list:
            for hidden in hidden_list:
                for bz in batch_list:
                    for l2 in l2_list:
                        for act in act_list:
                            t0 = time.perf_counter()
                            seed_results = []
                            for sd in seed_list:
                                trial_model = os.path.join(
                                    args.output_dir,
                                    f"grid_{lr}_{hidden}_{bz}_{l2}_{act}_seed{sd}.npy",
                                )
                                trial_hist = os.path.join(
                                    args.output_dir,
                                    f"history_{lr}_{hidden}_{bz}_{l2}_{act}_seed{sd}.csv",
                                )
                                val_acc, hist = run_train_with_params(
                                    lr_max=lr,
                                    hidden_dim=hidden,
                                    batch_size=bz,
                                    l2_lambda=l2,
                                    epochs=args.search_epochs,
                                    patience=args.search_patience,
                                    activation=act,
                                    seed=sd,
                                    model_path=trial_model,
                                    history_path=trial_hist,
                                    save_best=False,
                                    data_path=args.data_path,
                                    preloaded_train_data=(x_train, y_train),
                                    use_validation_split=True,
                                    val_ratio=args.val_ratio,
                                    split_seed=args.split_seed,
                                    min_delta=args.min_delta,
                                    grad_clip_norm=args.grad_clip_norm,
                                )
                                best_epoch = max(hist, key=lambda x: x["val_acc"])["epoch"] if hist else -1
                                seed_results.append(
                                    {
                                        "seed": int(sd),
                                        "best_val_acc": float(val_acc),
                                        "best_epoch": int(best_epoch),
                                    }
                                )
                            elapsed_sec = time.perf_counter() - t0
                            acc_values = np.array([r["best_val_acc"] for r in seed_results], dtype=np.float64)
                            val_acc_mean = float(np.mean(acc_values))
                            val_acc_std = float(np.std(acc_values))
                            best_seed_item = max(seed_results, key=lambda x: x["best_val_acc"])
                            writer.writerow(
                                [
                                    lr,
                                    hidden,
                                    bz,
                                    l2,
                                    act,
                                    len(seed_results),
                                    f"{val_acc_mean:.6f}",
                                    f"{val_acc_std:.6f}",
                                    f"{best_seed_item['best_val_acc']:.6f}",
                                    best_seed_item["best_epoch"],
                                    f"{elapsed_sec:.2f}",
                                    json.dumps(seed_results, ensure_ascii=False),
                                ]
                            )
                            print(
                                f"Log: {lr} {hidden} {bz} {l2} {act} | "
                                f"val_acc_mean={val_acc_mean:.4f} ± {val_acc_std:.4f} | "
                                f"best_single_seed={best_seed_item['seed']}:{best_seed_item['best_val_acc']:.4f} | "
                                f"{elapsed_sec:.2f}s"
                            )
                            all_trials.append(
                                {
                                    "lr": lr,
                                    "hidden": hidden,
                                    "batch": bz,
                                    "l2": l2,
                                    "activation": act,
                                    "seed_count": int(len(seed_results)),
                                    "val_acc_mean": val_acc_mean,
                                    "val_acc_std": val_acc_std,
                                    "best_val_acc_single_seed": float(best_seed_item["best_val_acc"]),
                                    "best_epoch_at_best_seed": int(best_seed_item["best_epoch"]),
                                    "best_seed": int(best_seed_item["seed"]),
                                    "seed_details": seed_results,
                                    "elapsed_sec": float(elapsed_sec),
                                }
                            )
                            if val_acc_mean > best_acc:
                                best_acc = val_acc_mean
                                best_cfg = {
                                    "lr": lr,
                                    "hidden": hidden,
                                    "batch": bz,
                                    "l2": l2,
                                    "activation": act,
                                    "best_seed": int(best_seed_item["seed"]),
                                    "best_val_acc_single_seed": float(best_seed_item["best_val_acc"]),
                                    "val_acc_mean": float(val_acc_mean),
                                    "val_acc_std": float(val_acc_std),
                                }

    if best_cfg is None:
        raise RuntimeError("Grid search did not produce any valid configuration.")

    lr = best_cfg["lr"]
    hidden = best_cfg["hidden"]
    bz = best_cfg["batch"]
    l2 = best_cfg["l2"]
    act = best_cfg["activation"]
    final_seed = best_cfg["best_seed"]
    print(
        f"Best config => lr={lr}, hidden={hidden}, batch={bz}, "
        f"l2={l2}, activation={act}, val_acc_mean={best_acc:.4f}, "
        f"final_seed(best_single_seed)={final_seed}"
    )
    top_trials = sorted(all_trials, key=lambda x: x["val_acc_mean"], reverse=True)[: args.topk]
    summary_dir = os.path.dirname(args.summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_config": {
                    "lr": lr,
                    "hidden": hidden,
                    "batch": bz,
                    "l2": l2,
                    "activation": act,
                    "final_training_seed": int(final_seed),
                    "best_val_acc_mean": float(best_acc),
                    "selection_rule": "max val_acc_mean over seed_list",
                    "seed_for_final_training_rule": "best single-seed val_acc within selected best_config",
                    "seed_list": seed_list,
                    "split_seed": int(args.split_seed),
                },
                "topk": int(args.topk),
                "top_trials": top_trials,
                "search_log_path": args.log_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Search summary saved to: {args.summary_path}")
    t0 = time.perf_counter()
    final_acc, final_hist = run_train_with_params(
        lr_max=lr,
        hidden_dim=hidden,
        batch_size=bz,
        l2_lambda=l2,
        epochs=args.final_epochs,
        patience=args.final_patience,
        activation=act,
        seed=final_seed,
        model_path=args.best_model_path,
        history_path=args.best_history_path,
        save_best=True,
        data_path=args.data_path,
        preloaded_train_data=(x_train, y_train),
        use_validation_split=True,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        min_delta=args.min_delta,
        grad_clip_norm=args.grad_clip_norm,
        best_meta_path=args.best_meta_path,
        config_path=args.best_config_path,
        split_path=args.split_path,
    )
    final_elapsed = time.perf_counter() - t0
    final_best_epoch = max(final_hist, key=lambda x: x["val_acc"])["epoch"] if final_hist else -1
    print(
        f"Final validation-based training finished. Best val score: {final_acc:.4f} | "
        f"best_epoch={final_best_epoch} | {final_elapsed:.2f}s"
    )

    if args.retrain_full_data:
        print("Retraining with full training set (train+val) using best hyper-parameters ...")
        t1 = time.perf_counter()
        full_score, full_hist = run_train_with_params(
            lr_max=lr,
            hidden_dim=hidden,
            batch_size=bz,
            l2_lambda=l2,
            epochs=args.final_epochs,
            patience=args.final_patience,
            activation=act,
            seed=final_seed,
            model_path=args.best_model_full_path,
            history_path=args.best_history_full_path,
            save_best=True,
            data_path=args.data_path,
            preloaded_train_data=(x_train, y_train),
            use_validation_split=False,
            val_ratio=0.0,
            min_delta=args.min_delta,
            grad_clip_norm=args.grad_clip_norm,
            best_meta_path=args.best_meta_full_path,
            config_path=args.best_config_full_path,
            split_path=None,
            save_final_if_no_val=True,
        )
        full_elapsed = time.perf_counter() - t1
        final_train_acc = full_hist[-1]["train_acc"] if full_hist else 0.0
        print(
            f"Full-data retraining finished. Final train acc={final_train_acc:.4f} | "
            f"tracked_score={full_score:.4f} | {full_elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()