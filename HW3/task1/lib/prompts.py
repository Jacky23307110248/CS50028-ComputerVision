"""Read text prompts from data/object_B and data/object_C."""

from __future__ import annotations

from pathlib import Path


class PromptFiles:
    def __init__(self, root: Path):
        self.root = root
        self.prompt_path = root / "prompt.txt"
        self.negative_path = root / "negative_prompt.txt"

    def require_prompt(self) -> str:
        if not self.prompt_path.is_file():
            raise FileNotFoundError(
                f"Missing {self.prompt_path}. Create it with a one-line English prompt."
            )
        text = self.prompt_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{self.prompt_path} is empty.")
        return text

    def negative_prompt(self) -> str | None:
        if not self.negative_path.is_file():
            return None
        text = self.negative_path.read_text(encoding="utf-8").strip()
        return text or None
