import argparse
import asyncio

from _orchestrator import JobSpec, add_scheduler_args, run_job_specs


def build_jobs(seed: int) -> list[JobSpec]:
    return [
        JobSpec(
            name="baseline_resnet18",
            config="baseline_resnet18.yaml",
            exp_name="baseline_resnet18",
            seed=seed,
            env_overrides={"T1_PRETRAINED": "1", "T1_EPOCHS": "20", "T1_ATTENTION": "none"},
        ),
        JobSpec(
            name="ablation_pretrained_false",
            config="ablation.yaml",
            exp_name="ablation_pretrained_false",
            seed=seed,
            env_overrides={"T1_PRETRAINED": "0", "T1_EPOCHS": "20", "T1_ATTENTION": "none"},
        ),
        JobSpec(
            name="ablation_scratch_ep80",
            config="ablation.yaml",
            exp_name="ablation_scratch_ep80",
            seed=seed,
            env_overrides={"T1_PRETRAINED": "0", "T1_EPOCHS": "80", "T1_ATTENTION": "none"},
        ),
        JobSpec(
            name="ablation_scratch_ep120",
            config="ablation.yaml",
            exp_name="ablation_scratch_ep120",
            seed=seed,
            env_overrides={"T1_PRETRAINED": "0", "T1_EPOCHS": "120", "T1_ATTENTION": "none"},
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run baseline + scratch ablation suite for Task1.")
    parser.add_argument("--seed", type=int, default=42)
    add_scheduler_args(parser)
    args = parser.parse_args()
    code = asyncio.run(run_job_specs(build_jobs(args.seed), args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
