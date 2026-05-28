"""Action normalization stats from checkpoint preprocessor (matches training)."""

from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file


def load_action_std_from_checkpoint(checkpoint_dir: Path) -> list[float]:
    """MEAN_STD action std saved with ACT pretrained_model."""
    ckpt = Path(checkpoint_dir)
    candidates = sorted(ckpt.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No normalizer safetensors under {ckpt}")
    state = load_file(str(candidates[0]))
    # Keys vary slightly across versions; match action std tensor.
    for key, tensor in state.items():
        if "action" in key and "std" in key:
            return tensor.flatten().tolist()
    raise KeyError(f"action std not found in {candidates[0]}")
