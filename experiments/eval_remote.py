"""
Evaluate all (or a subset of) environments against a pre-deployed policy server.

IsaacLab can only be initialised once per process, so each environment is
evaluated in its own subprocess via scripts/eval.py.

Usage:
    # Evaluate all environments against a server at 10.0.0.5:8000
    uv run python experiments/eval_remote.py --host 10.0.0.5 --port 8000

    # Only specific environments, 10 rollouts each
    uv run python experiments/eval_remote.py --host 10.0.0.5 --port 8000 \
        --envs DROID-FoodBussing DROID-PanClean --rollouts 10

    # Custom output directory and open-loop horizon
    uv run python experiments/eval_remote.py --host 10.0.0.5 --port 8000 \
        --output-dir runs/pi0_fast_experiment --open-loop-horizon 4

    # Dry-run: print the commands without executing
    uv run python experiments/eval_remote.py --host 10.0.0.5 --port 8000 --dry-run
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tyro

from polaris.config import EvalArgs, PolicyArgs

ALL_ENVIRONMENTS = [
    "DROID-BlockStackKitchen",
    "DROID-FoodBussing",
    "DROID-PanClean",
    "DROID-MoveLatteCup",
    "DROID-OrganizeTools",
    "DROID-TapeIntoContainer",
]


@dataclass
class ExperimentArgs:
    """Configure an evaluation sweep against a running policy server."""
    exp_name: str
    host: str = '0.0.0.0'
    port: int = 8000
    client: str = "DroidJointPos"
    open_loop_horizon: int = 8
    envs: list[str] = field(default_factory=lambda: ALL_ENVIRONMENTS)
    rollouts: int | None = None
    output_dir: str = "runs"
    headless: bool = True
    dry_run: bool = False


def build_eval_jobs(exp: ExperimentArgs) -> list[EvalArgs]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy = PolicyArgs(
        client=exp.client,
        host=exp.host,
        port=exp.port,
        open_loop_horizon=exp.open_loop_horizon,
    )
    jobs: list[EvalArgs] = []
    for env_name in exp.envs:
        jobs.append(
            EvalArgs(
                policy=policy,
                environment=env_name,
                run_folder=str(Path(exp.output_dir) / exp.exp_name / (env_name + "_" + timestamp)),
                headless=exp.headless,
                rollouts=exp.rollouts,
            )
        )
    return jobs


def eval_args_to_cli(args: EvalArgs) -> list[str]:
    """Serialize an EvalArgs into the tyro CLI flags that scripts/eval.py expects."""
    cli = [
        "--policy.client", args.policy.client,
        "--policy.host", args.policy.host,
        "--policy.port", str(args.policy.port),
        "--environment", args.environment,
        "--run-folder", args.run_folder,
    ]
    if args.policy.open_loop_horizon is not None:
        cli += ["--policy.open-loop-horizon", str(args.policy.open_loop_horizon)]
    if args.headless:
        cli += ["--headless"]
    else:
        cli += ["--no-headless"]
    if args.initial_conditions_file is not None:
        cli += ["--initial-conditions-file", args.initial_conditions_file]
    if args.instruction is not None:
        cli += ["--instruction", args.instruction]
    if args.rollouts is not None:
        cli += ["--rollouts", str(args.rollouts)]
    return cli


def main() -> None:
    exp = tyro.cli(ExperimentArgs)
    jobs = build_eval_jobs(exp)

    eval_script = str(Path(__file__).resolve().parent.parent / "scripts" / "eval.py")

    print(f"╔══ Evaluation sweep: {len(jobs)} job(s) ══")
    print(f"║ Policy server : {exp.host}:{exp.port}")
    print(f"║ Client        : {exp.client}")
    print(f"║ Environments  : {exp.envs}")
    print(f"║ Output dir    : {exp.output_dir}")
    if exp.dry_run:
        print(f"║ ** DRY RUN — commands will be printed, not executed **")
    print(f"╚{'═' * 50}")
    print()

    failures: list[str] = []
    for i, job in enumerate(jobs):
        cli_args = eval_args_to_cli(job)
        cmd = [sys.executable, eval_script] + cli_args

        print(f"[{i + 1}/{len(jobs)}] {job.environment}")
        print(f"     run_folder: {job.run_folder}")
        print(f"     $ {' '.join(cmd)}")

        if exp.dry_run:
            print()
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"     FAILED (exit code {result.returncode})")
            failures.append(job.environment)
        else:
            print(f"     DONE")
        print()

    if not exp.dry_run:
        print(f"=== Sweep complete: {len(jobs) - len(failures)}/{len(jobs)} succeeded ===")
        if failures:
            print(f"    Failed: {failures}")


if __name__ == "__main__":
    main()
