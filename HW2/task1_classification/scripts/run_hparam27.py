import argparse
import asyncio

from _orchestrator import JobSpec, add_scheduler_args, run_job_specs


def build_jobs(seed: int) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for blr in [0.0001, 0.0002, 0.0003]:
        for hlr in [0.001, 0.002, 0.003]:
            for ep in [20, 30, 40]:
                exp = f"hparam27_blr{blr:.4f}_hlr{hlr:.3f}_ep{ep}"
                jobs.append(
                    JobSpec(
                        name=exp,
                        config="baseline_resnet18.yaml",
                        exp_name=exp,
                        seed=seed,
                        env_overrides={
                            "T1_BACKBONE_LR": str(blr),
                            "T1_HEAD_LR": str(hlr),
                            "T1_EPOCHS": str(ep),
                            "T1_PRETRAINED": "1",
                            "T1_ATTENTION": "none",
                        },
                    )
                )
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run 27 hyperparameter search jobs for Task1.")
    parser.add_argument("--seed", type=int, default=42)
    add_scheduler_args(parser)
    args = parser.parse_args()
    code = asyncio.run(run_job_specs(build_jobs(args.seed), args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
