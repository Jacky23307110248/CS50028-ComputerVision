#!/usr/bin/env python
"""Merge eval_summary.json files into one ID/OOD comparison table (CSV)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT))


def load_summary(path: Path) -> dict:
    return json.loads((path / "eval_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Pairs: label=path_to_eval_output_dir",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HW3_ROOT / "task2" / "outputs" / "eval_compare_id_ood.csv",
    )
    args = parser.parse_args()

    rows = []
    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"Expected label=dir, got {spec!r}")
        label, dir_path = spec.split("=", 1)
        s = load_summary(Path(dir_path))
        rows.append(
            {
                "label": label,
                "model_tag": s.get("model_tag"),
                "split": s.get("split"),
                "l1_norm": s.get("l1_norm"),
                "mae_denorm": s.get("mae_denorm"),
                "l1_pos": s.get("l1_pos"),
                "l1_rot": s.get("l1_rot"),
                "l1_grip": s.get("l1_grip"),
                "baseline_zero_l1": s.get("baseline_zero_l1"),
                "margin_vs_zero_pct": s.get("margin_vs_zero_pct"),
                "l1_ep_median": s.get("l1_ep_median"),
                "l1_chunk_first_half": s.get("l1_chunk_first_half"),
                "l1_chunk_second_half": s.get("l1_chunk_second_half"),
                "num_episodes": s.get("num_episodes"),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
