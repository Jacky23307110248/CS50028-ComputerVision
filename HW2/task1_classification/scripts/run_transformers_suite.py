import argparse
import asyncio

from _orchestrator import JobSpec, add_scheduler_args, run_job_specs


def build_jobs(seed: int) -> list[JobSpec]:
    return [
        JobSpec(
            name="baseline_resnet18_aligned_transformer_adamw",
            config="baseline_resnet18_aligned_transformer.yaml",
            exp_name="baseline_resnet18_aligned_transformer_adamw",
            seed=seed,
        ),
        JobSpec(
            name="vit_tiny_adamw",
            config="vit_tiny.yaml",
            exp_name="vit_tiny_adamw",
            seed=seed,
        ),
        JobSpec(
            name="swin_tiny_adamw",
            config="swin_tiny.yaml",
            exp_name="swin_tiny_adamw",
            seed=seed,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run transformer-aligned Task1 suite (30 epochs).")
    parser.add_argument("--seed", type=int, default=42)
    add_scheduler_args(parser)
    args = parser.parse_args()
    code = asyncio.run(run_job_specs(build_jobs(args.seed), args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
