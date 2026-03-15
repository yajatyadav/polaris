"""
Run ONE eval job: N parallel env evals querying a remote policy server.

No server is launched here — the policy server runs on a separate machine.
This script just runs eval subprocesses that query the remote server.

Each job evaluates a specific num_candidates value across a set of envs.

Usage:
    uv run python experiments/run_classifier_guided_job.py \
        --host 10.0.0.1 --port 8000 \
        --num-candidates 64 \
        --envs DROID-FoodBussing DROID-PanClean DROID-MoveLatteCup \
        --rollouts 5

    # Dry-run: print commands without executing
    uv run python experiments/run_classifier_guided_job.py ... --dry-run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
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


def _eval_args_to_cli(args: EvalArgs) -> list[str]:
    cli = [
        "--policy.client", args.policy.client,
        "--policy.host", args.policy.host,
        "--policy.port", str(args.policy.port),
        "--environment", args.environment,
        "--run-folder", args.run_folder,
    ]
    if args.policy.num_candidates is not None:
        cli += ["--policy.num-candidates", str(args.policy.num_candidates)]
    if args.policy.open_loop_horizon is not None:
        cli += ["--policy.open-loop-horizon", str(args.policy.open_loop_horizon)]
    cli += ["--headless"] if args.headless else ["--no-headless"]
    if args.initial_conditions_file is not None:
        cli += ["--initial-conditions-file", args.initial_conditions_file]
    if args.instruction is not None:
        cli += ["--instruction", args.instruction]
    if args.rollouts is not None:
        cli += ["--rollouts", str(args.rollouts)]
    return cli


@dataclass
class ClassifierGuidedJobArgs:
    """One job = list of env evals querying a remote policy server."""

    # --- Remote policy server ---
    host: str = "0.0.0.0"
    port: int = 8100
    num_candidates: int | None = None  # Must be explicitly set

    # --- Eval (scripts/eval.py) ---
    client: str = "DroidJointPos"
    open_loop_horizon: int = 8
    rollouts: int | None = None
    output_dir: str = "runs"
    job_name: str | None = None
    headless: bool = True
    initial_conditions_file: str | None = None
    instruction: str | None = None

    # --- Cluster / batch run ---
    run_dir_name: str | None = None
    job_id: int = 0
    intermediate_base_dir: str = "runs/classifier_guided_jobs"

    # --- Envs to evaluate ---
    envs: list[str] = field(default_factory=lambda: ALL_ENVIRONMENTS)

    # --- GPU & resource ---
    gpu_id: int = 0
    eval_mem_total: float = 0.95

    # --- Control ---
    dry_run: bool = False


def main() -> None:
    args = tyro.cli(ClassifierGuidedJobArgs)

    if args.num_candidates is None:
        raise ValueError("--num-candidates must be explicitly set")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_dir_name is not None:
        job_output_root = (
            Path(args.intermediate_base_dir)
            / args.run_dir_name
            / "job_results"
            / f"job_{args.job_id:03d}"
            / f"N{args.num_candidates}"
        )
        job_label = f"job_{args.job_id:03d}"
        run_folder_tmpl = str(job_output_root / "{env}_{ts}")
    else:
        job_label = args.job_name or timestamp
        run_folder_tmpl = str(Path(args.output_dir) / job_label / "{env}_{ts}")

    policy = PolicyArgs(
        client=args.client,
        host=args.host,
        port=args.port,
        open_loop_horizon=args.open_loop_horizon,
        num_candidates=args.num_candidates,
    )
    eval_jobs: list[EvalArgs] = []
    for env_name in args.envs:
        run_folder = run_folder_tmpl.format(env=env_name, ts=timestamp)
        eval_jobs.append(
            EvalArgs(
                policy=policy,
                environment=env_name,
                run_folder=run_folder,
                headless=args.headless,
                rollouts=args.rollouts,
                initial_conditions_file=args.initial_conditions_file,
                instruction=args.instruction,
            )
        )

    num_evals = len(eval_jobs)
    eval_mem_each = args.eval_mem_total / num_evals if num_evals > 0 else 0

    eval_script = "scripts/eval.py"

    output_display = (
        f"{args.intermediate_base_dir}/{args.run_dir_name}/job_results/{job_label}/N{args.num_candidates}"
        if args.run_dir_name is not None
        else f"{args.output_dir}/{job_label}"
    )
    print("╔══ Classifier-guided eval job ══")
    print(f"║ Remote server : {args.host}:{args.port}")
    print(f"║ num_candidates: {args.num_candidates}")
    print(f"║ Evals         : {num_evals} envs, {eval_mem_each*100:.1f}% GPU mem each")
    print(f"║ Envs          : {args.envs}")
    print(f"║ Output        : {output_display}")
    if args.dry_run:
        print("║ ** DRY RUN **")
    print("╚══════════════════════════════")
    print()

    polaris_root = Path(__file__).resolve().parent.parent
    eval_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "POLARIS_CUDA_MEMORY_FRACTION": str(eval_mem_each),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OMP_NUM_THREADS": "1",
    }

    if args.dry_run:
        for i, job in enumerate(eval_jobs):
            cli = _eval_args_to_cli(job)
            print(f"Eval [{i+1}] {job.environment}:")
            print(f"  uv run python {eval_script} {' '.join(cli)}")
        return

    # Save job metadata when running in cluster mode
    if args.run_dir_name is not None:
        job_output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_dir_name": args.run_dir_name,
            "job_id": args.job_id,
            "timestamp": timestamp,
            "host": args.host,
            "port": args.port,
            "num_candidates": args.num_candidates,
            "envs": args.envs,
            "rollouts": args.rollouts,
        }
        (job_output_root / "job_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Launch all evals in parallel
    print(f"Launching {num_evals} eval processes in parallel...")

    def _prefix_stream(stream, prefix: str, dest):
        """Read lines from stream and write them to dest with a prefix."""
        for line in stream:
            dest.write(f"{prefix} {line}")
            dest.flush()
        stream.close()

    procs: list[tuple[str, subprocess.Popen]] = []
    threads: list[threading.Thread] = []
    for i, job in enumerate(eval_jobs):
        cli = _eval_args_to_cli(job)
        cmd = ["uv", "run", "python", "-u", eval_script] + cli
        p = subprocess.Popen(
            cmd, cwd=polaris_root, env=eval_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        prefix = f"[{i}]"
        t_out = threading.Thread(target=_prefix_stream, args=(p.stdout, prefix, sys.stdout), daemon=True)
        t_err = threading.Thread(target=_prefix_stream, args=(p.stderr, prefix, sys.stderr), daemon=True)
        t_out.start()
        t_err.start()
        threads.extend([t_out, t_err])
        procs.append((job.environment, p))

    # Wait for all
    failures: list[str] = []
    for env_name, p in procs:
        p.wait()
        if p.returncode != 0:
            failures.append(env_name)
            print(f"  [{env_name}] FAILED (exit {p.returncode})")
        else:
            print(f"  [{env_name}] DONE")
    for t in threads:
        t.join(timeout=5)

    print()
    print(f"=== Job complete: {num_evals - len(failures)}/{num_evals} evals succeeded ===")
    if failures:
        print(f"    Failed: {failures}")


if __name__ == "__main__":
    main()
