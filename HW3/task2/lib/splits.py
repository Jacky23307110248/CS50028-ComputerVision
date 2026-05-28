"""Train / val / test episode splits. Environment D is test-only (never train/val)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR, SPLITS_PATH, env_dataset_root

TRAIN_ENVS = ("A", "B", "C")
TEST_ENV = "D"
ALL_ENVS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class SplitSpec:
    seed: int
    val_ratio: float
    train: dict[str, list[int]]
    val: dict[str, list[int]]
    test: dict[str, list[int]]

    def train_episodes(self, env: str) -> list[int]:
        return list(self.train[env])

    def val_episodes(self, env: str) -> list[int]:
        return list(self.val[env])

    def test_episodes(self, env: str = TEST_ENV) -> list[int]:
        return list(self.test[env])

    def assert_no_leakage(self) -> None:
        """Hard checks: disjoint splits; D never in train/val."""
        if TEST_ENV in self.train or TEST_ENV in self.val:
            raise ValueError(f"{TEST_ENV} must not appear in train/val splits")

        for env in TRAIN_ENVS:
            tr = set(self.train[env])
            va = set(self.val[env])
            te = set(self.test.get(env, []))
            if tr & va:
                raise ValueError(f"Env {env}: train/val overlap ({len(tr & va)} episodes)")
            if tr & te or va & te:
                raise ValueError(f"Env {env}: train/val overlaps test")
            if env in self.test:
                raise ValueError(f"Only {TEST_ENV} may be in test dict, found {env}")

        # Episode indices are per-environment (0..N-1), not global IDs.
        # Leakage is prevented by never loading env D in train/val builders, not by index equality.


def _episode_count(env: str) -> int:
    """Read episode count from meta/info.json (works before v2.1→v3.0 conversion)."""
    import json

    info_path = env_dataset_root(env) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["total_episodes"])


def build_splits(seed: int = 42, val_ratio: float = 0.05) -> SplitSpec:
    """Build deterministic per-env train/val; all D episodes go to test only."""
    rng = random.Random(seed)
    train: dict[str, list[int]] = {}
    val: dict[str, list[int]] = {}
    test: dict[str, list[int]] = {}

    for env in TRAIN_ENVS:
        n = _episode_count(env)
        indices = list(range(n))
        rng.shuffle(indices)
        n_val = max(1, int(round(n * val_ratio)))
        val_idx = sorted(indices[:n_val])
        train_idx = sorted(indices[n_val:])
        train[env] = train_idx
        val[env] = val_idx

    n_d = _episode_count(TEST_ENV)
    test[TEST_ENV] = list(range(n_d))

    spec = SplitSpec(seed=seed, val_ratio=val_ratio, train=train, val=val, test=test)
    spec.assert_no_leakage()
    return spec


def save_splits(spec: SplitSpec, path: Path | None = None) -> Path:
    path = path or SPLITS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "seed": spec.seed,
        "val_ratio": spec.val_ratio,
        "policy": {
            "train_envs": list(TRAIN_ENVS),
            "test_env": TEST_ENV,
            "note": "D is test-only. Never use test episodes for training or validation.",
        },
        "train": spec.train,
        "val": spec.val,
        "test": spec.test,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_splits(path: Path | None = None) -> SplitSpec:
    path = path or SPLITS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python task2/scripts/build_splits.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    spec = SplitSpec(
        seed=int(data["seed"]),
        val_ratio=float(data["val_ratio"]),
        train={k: list(v) for k, v in data["train"].items()},
        val={k: list(v) for k, v in data["val"].items()},
        test={k: list(v) for k, v in data["test"].items()},
    )
    spec.assert_no_leakage()
    return spec
