import argparse
import asyncio

from _orchestrator import JobSpec, add_scheduler_args, run_job_specs


def _job(seed: int, exp_name: str, attention: str, epochs: int) -> JobSpec:
    return JobSpec(
        name=exp_name,
        config="baseline_resnet18.yaml",
        exp_name=exp_name,
        seed=seed,
        env_overrides={
            "T1_ATTENTION": attention,
            "T1_EPOCHS": str(epochs),
            "T1_PRETRAINED": "1",
        },
    )


def build_jobs(seed: int) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    jobs.extend(
        [
            _job(seed, "baseline_resnet18", "none", 20),
            _job(seed, "attention_se", "se", 20),
            _job(seed, "attention_cbam", "cbam", 20),
            _job(seed, "attention_se_high", "se_high", 20),
            _job(seed, "attention_cbam_high", "cbam_high", 20),
        ]
    )
    jobs.extend(
        [
            _job(seed, "baseline_no_attention_ep40", "none", 40),
            _job(seed, "attention_se_ep40", "se", 40),
            _job(seed, "attention_cbam_ep40", "cbam", 40),
            _job(seed, "attention_se_high_ep40", "se_high", 40),
            _job(seed, "attention_cbam_high_ep40", "cbam_high", 40),
        ]
    )
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run attention block suite for Task1 (20ep + 40ep).")
    parser.add_argument("--seed", type=int, default=42)
    add_scheduler_args(parser)
    args = parser.parse_args()
    code = asyncio.run(run_job_specs(build_jobs(args.seed), args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
