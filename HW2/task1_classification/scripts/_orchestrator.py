import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "runner.py"


@dataclass
class JobSpec:
    name: str
    config: str
    exp_name: str
    seed: int = 42
    env_overrides: Optional[Dict[str, str]] = None
    needs_gpu: bool = True


def add_scheduler_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=12,
        help="Max total concurrent jobs (default tuned for ~23 CPU cores).",
    )
    parser.add_argument(
        "--max-gpu-jobs",
        type=int,
        default=8,
        help="Max concurrent GPU jobs (default tuned for a 192GB GPU).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print jobs only, do not execute.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop scheduling new jobs after first failure.")
    return parser


def _build_command(job: JobSpec) -> List[str]:
    cfg_path = ROOT / "configs" / job.config
    return [
        sys.executable,
        str(RUNNER_PATH),
        "--config",
        str(cfg_path),
        "--override-exp-name",
        job.exp_name,
        "--override-seed",
        str(job.seed),
    ]


async def _run_one_job(
    idx: int,
    total: int,
    job: JobSpec,
    sem_parallel: asyncio.Semaphore,
    sem_gpu: asyncio.Semaphore,
) -> tuple[str, int, float]:
    async with sem_parallel:
        if job.needs_gpu:
            async with sem_gpu:
                return await _spawn_job(idx, total, job)
        return await _spawn_job(idx, total, job)


async def _spawn_job(idx: int, total: int, job: JobSpec) -> tuple[str, int, float]:
    cmd = _build_command(job)
    env = os.environ.copy()
    if job.env_overrides:
        env.update(job.env_overrides)

    start = time.time()
    print(f"[{idx}/{total}] START {job.name}: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(ROOT),
        env=env,
    )
    code = await proc.wait()
    elapsed = time.time() - start
    status = "OK" if code == 0 else "FAIL"
    print(f"[{idx}/{total}] {status} {job.name} (exit={code}, {elapsed:.1f}s)")
    return job.name, code, elapsed


async def run_job_specs(jobs: Sequence[JobSpec], args: argparse.Namespace) -> int:
    if not jobs:
        print("No jobs to run.")
        return 0

    print(f"Total jobs: {len(jobs)}")
    print(f"Scheduler: max_parallel_jobs={args.max_parallel_jobs}, max_gpu_jobs={args.max_gpu_jobs}")
    for i, j in enumerate(jobs, start=1):
        env_text = "" if not j.env_overrides else f" env={j.env_overrides}"
        print(f"  - [{i}] {j.name}: config={j.config}, exp={j.exp_name}{env_text}")

    if args.dry_run:
        return 0

    sem_parallel = asyncio.Semaphore(max(1, args.max_parallel_jobs))
    sem_gpu = asyncio.Semaphore(max(1, args.max_gpu_jobs))

    tasks = []
    for idx, job in enumerate(jobs, start=1):
        tasks.append(asyncio.create_task(_run_one_job(idx, len(jobs), job, sem_parallel, sem_gpu)))

    results: List[tuple[str, int, float]] = []
    if args.fail_fast:
        for t in asyncio.as_completed(tasks):
            name, code, elapsed = await t
            results.append((name, code, elapsed))
            if code != 0:
                print(f"Fail-fast triggered by {name}.")
                for p in tasks:
                    if not p.done():
                        p.cancel()
                break
    else:
        results = await asyncio.gather(*tasks)

    failed = [r for r in results if r[1] != 0]
    print("\n===== Summary =====")
    print(f"Finished: {len(results)}/{len(jobs)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for name, code, elapsed in failed:
            print(f"  - {name}: exit={code}, elapsed={elapsed:.1f}s")
        return 1
    return 0
