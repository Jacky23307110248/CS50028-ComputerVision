"""SwanLab + JSONL logging for task1 wrapper scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task1.lib.paths import SWANLAB_PROJECT, SWANLAB_WORKSPACE


class RunLogger:
    def __init__(
        self,
        run_name: str,
        log_dir: Path,
        *,
        use_swanlab: bool = True,
        config: dict[str, Any] | None = None,
    ):
        self.run_name = run_name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.use_swanlab = use_swanlab
        self._swan = None
        if use_swanlab:
            try:
                import swanlab

                self._swan = swanlab
                kwargs: dict[str, Any] = {
                    "project": SWANLAB_PROJECT,
                    "workspace": SWANLAB_WORKSPACE,
                    "experiment_name": run_name,
                    "logdir": str(log_dir),
                }
                if config:
                    kwargs["config"] = config
                self._swan.init(**kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] SwanLab init failed ({exc}); JSONL only.")
                self.use_swanlab = False

    def log(self, step: int, metrics: dict[str, float | int | str]) -> None:
        row: dict[str, Any] = {"step": step, **metrics}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.use_swanlab and self._swan is not None:
            numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if numeric:
                self._swan.log(numeric, step=step)

    def finish(self) -> None:
        if self.use_swanlab and self._swan is not None:
            self._swan.finish()
