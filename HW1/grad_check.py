import argparse
import json
import os
import numpy as np

from dataloader import load_mnist
from model import Tape, mlp_logits, stable_softmax_ce, l2_loss, get_activation
from utils import he_init, xavier_init


INPUT_DIM = 784
OUT_DIM = 10


def compute_loss(x, y, w1, b1, w2, b2, activation_fn, l2_lambda):
    logits = mlp_logits(x, w1, b1, w2, b2, activation_fn)
    ce_loss, _ = stable_softmax_ce(logits, y)
    return float(ce_loss + l2_loss(w1, w2, l2_lambda))


def compute_analytic_grads(x, y, w1, b1, w2, b2, activation_fn, l2_lambda):
    tape = Tape()
    logits = mlp_logits(x, w1, b1, w2, b2, activation_fn, tape=tape)
    ce_loss, _ = stable_softmax_ce(logits, y, tape=tape)
    tape.backward(ce_loss, grad_out=1.0)
    dw1 = tape.get_grad(w1) + l2_lambda * w1
    db1 = tape.get_grad(b1)
    dw2 = tape.get_grad(w2) + l2_lambda * w2
    db2 = tape.get_grad(b2)
    return dw1, db1, dw2, db2


def numerical_grad_for_param(param, index, loss_fn, eps):
    old = param[index]
    param[index] = old + eps
    loss_plus = loss_fn()
    param[index] = old - eps
    loss_minus = loss_fn()
    param[index] = old
    return (loss_plus - loss_minus) / (2.0 * eps)


def rel_error(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def sample_indices(rng, shape, k):
    return [tuple(rng.integers(0, s) for s in shape) for _ in range(k)]


def build_args():
    parser = argparse.ArgumentParser(description="Numerical gradient check for MLP")
    parser.add_argument("--data_path", type=str, default="./FashionMNIST/raw")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid"])
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--checks_per_param", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--save_json_path",
        type=str,
        default="./ExtraData/grad_check_result.json",
        help="Path to save gradient-check details as JSON. Set empty string to disable.",
    )
    return parser.parse_args()


def main():
    args = build_args()
    if args.hidden_dim <= 0 or args.batch_size <= 0 or args.checks_per_param <= 0:
        raise ValueError("hidden_dim, batch_size and checks_per_param must be > 0")
    if args.l2 < 0 or args.eps <= 0 or args.tol <= 0:
        raise ValueError("l2 must be >= 0, eps and tol must be > 0")

    rng = np.random.default_rng(args.seed)
    activation_fn = get_activation(args.activation)
    dtype = np.float64 if args.dtype == "float64" else np.float32

    x_train, y_train, _, _, _, _ = load_mnist(data_path=args.data_path)
    x = x_train[: args.batch_size].astype(dtype)
    y = y_train[: args.batch_size].astype(dtype)

    if args.activation == "relu":
        w1 = he_init(INPUT_DIM, args.hidden_dim, rng=rng)
        w2 = he_init(args.hidden_dim, OUT_DIM, rng=rng)
    else:
        w1 = xavier_init(INPUT_DIM, args.hidden_dim, rng=rng)
        w2 = xavier_init(args.hidden_dim, OUT_DIM, rng=rng)
    w1 = w1.astype(dtype)
    w2 = w2.astype(dtype)
    b1 = np.zeros(args.hidden_dim, dtype=dtype)
    b2 = np.zeros(OUT_DIM, dtype=dtype)

    dw1, db1, dw2, db2 = compute_analytic_grads(
        x, y, w1, b1, w2, b2, activation_fn, args.l2
    )

    def current_loss():
        return compute_loss(x, y, w1, b1, w2, b2, activation_fn, args.l2)

    checks = [
        ("W1", w1, dw1),
        ("b1", b1, db1),
        ("W2", w2, dw2),
        ("b2", b2, db2),
    ]

    all_errors = []
    detailed_results = {}
    for name, param, grad in checks:
        idx_list = sample_indices(rng, param.shape, args.checks_per_param)
        print(f"\nChecking {name} ({len(idx_list)} points)")
        detailed_results[name] = []
        for idx in idx_list:
            g_num = numerical_grad_for_param(param, idx, current_loss, args.eps)
            g_ana = float(grad[idx])
            err = rel_error(g_num, g_ana)
            all_errors.append(err)
            detailed_results[name].append(
                {
                    "idx": [int(i) for i in idx],
                    "g_num": float(g_num),
                    "g_ana": float(g_ana),
                    "rel_err": float(err),
                }
            )
            print(
                f"  idx={idx} | g_num={g_num:.6e} | g_ana={g_ana:.6e} | rel_err={err:.6e}"
            )

    max_err = max(all_errors) if all_errors else 0.0
    mean_err = float(np.mean(all_errors)) if all_errors else 0.0
    print(f"\nGradient check done. max_rel_err={max_err:.6e}, mean_rel_err={mean_err:.6e}")
    passed = max_err <= args.tol
    if passed:
        print("PASS")
    else:
        print("FAIL")

    if args.save_json_path:
        save_dir = os.path.dirname(args.save_json_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        payload = {
            "settings": {
                "data_path": args.data_path,
                "hidden_dim": int(args.hidden_dim),
                "batch_size": int(args.batch_size),
                "activation": args.activation,
                "l2": float(args.l2),
                "eps": float(args.eps),
                "checks_per_param": int(args.checks_per_param),
                "seed": int(args.seed),
                "tol": float(args.tol),
                "dtype": args.dtype,
            },
            "random_sampling": {
                "generator": "numpy.default_rng",
                "seed": int(args.seed),
                "per_parameter_points": int(args.checks_per_param),
            },
            "details": detailed_results,
            "summary": {
                "total_checks": int(len(all_errors)),
                "max_rel_err": float(max_err),
                "mean_rel_err": float(mean_err),
                "status": "PASS" if passed else "FAIL",
            },
        }
        with open(args.save_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"JSON saved to: {args.save_json_path}")


if __name__ == "__main__":
    main()
