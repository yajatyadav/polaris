"""
Run ONE job: one classifier-guided policy server + N parallel evals on the same GPU.

Starts the policy server in the background (35% GPU mem), waits for it to be ready,
then launches eval scripts in parallel (60% / N mem each). All run on the same GPU.

Usage:
    uv run python experiments/run_classifier_guided_job.py \\
        --actor-config pi05_droid_jointpos_polaris \\
        --actor-dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris \\
        --classifier-config polaris_droid_language_classifier_joint_pos \\
        --classifier-dir checkpoints/polaris_droid_language_classifier_joint_pos/polaris_droid_joint_pos_lang_clsfr__filteridle_lr5e-4_sgd_emaNone_batch32__val0.05x20/2000 \\
        --envs DROID-FoodBussing DROID-PanClean DROID-MoveLatteCup \\
        --rollouts 5

    # Dry-run: print commands without executing
    uv run python experiments/run_classifier_guided_job.py ... --dry-run
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout_sec: float = 120) -> bool:
    """Poll /healthz until server responds or timeout."""
    url = f"http://{host}:{port}/healthz"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as _:
                return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    return False


def _eval_args_to_cli(args: EvalArgs) -> list[str]:
    cli = [
        "--policy.client", args.policy.client,
        "--policy.host", args.policy.host,
        "--policy.port", str(args.policy.port),
        "--environment", args.environment,
        "--run-folder", args.run_folder,
    ]
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
    """One job = one policy server + list of env evals. All params are passable."""

    # --- Policy server (serve_classifier_guided_policy.py) ---
    actor_config: str
    actor_dir: str
    classifier_config: str
    classifier_dir: str
    num_candidates: int = 64
    default_prompt: str | None = None
    server_mem_fraction: float = 0.35

    # --- Eval (scripts/eval.py) ---
    client: str = "DroidJointPos"
    open_loop_horizon: int = 8
    rollouts: int | None = None
    output_dir: str = "runs"
    job_name: str | None = None  # If None, uses timestamp
    headless: bool = True
    initial_conditions_file: str | None = None
    instruction: str | None = None

    # --- Cluster / batch run (for sbatch + postprocess workflow) ---
    run_dir_name: str | None = None  # e.g. "my_run_20260304_123456"
    job_id: int = 0  # Unique per sbatch job when multiple jobs in one run
    intermediate_base_dir: str = "runs/classifier_guided_jobs"

    # --- Envs to evaluate (one EvalArgs per env) ---
    envs: list[str] = field(default_factory=lambda: ALL_ENVIRONMENTS)

    # --- GPU & resource ---
    gpu_id: int = 0
    eval_mem_total: float = 0.6  # Total GPU fraction for all evals (split equally)

    # --- Control ---
    dry_run: bool = False
    server_ready_timeout: float = 120.0


def main() -> None:
    args = tyro.cli(ClassifierGuidedJobArgs)
    port = _find_free_port()
    host = "127.0.0.1"

    # Build EvalArgs for each env
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use cluster run dir when provided; else legacy output_dir/job_name
    if args.run_dir_name is not None:
        job_output_root = (
            Path(args.intermediate_base_dir)
            / args.run_dir_name
            / "job_results"
            / f"job_{args.job_id:03d}"
        )
        job_label = f"job_{args.job_id:03d}"
        # Unique per-env folder: env + timestamp avoids duplicates across retries
        run_folder_tmpl = str(job_output_root / "{env}_{ts}")
    else:
        job_label = args.job_name or timestamp
        run_folder_tmpl = str(Path(args.output_dir) / job_label / "{env}_{ts}")

    policy = PolicyArgs(
        client=args.client,
        host=host,
        port=port,
        open_loop_horizon=args.open_loop_horizon,
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

    # All paths relative to polaris root (assume cwd is polaris when running)
    openpi_dir = "third_party/openpi"
    eval_script = "scripts/eval.py"

    # Build policy server command (tyro uses kebab-case for CLI)
    # Run from third_party/openpi so "scripts/serve_classifier_guided_policy.py" resolves
    server_cmd = [
        "uv", "run", "python", "scripts/serve_classifier_guided_policy.py",
        "--actor-config", args.actor_config,
        "--actor-dir", args.actor_dir,
        "--classifier-config", args.classifier_config,
        "--classifier-dir", args.classifier_dir,
        "--num-candidates", str(args.num_candidates),
        "--port", str(port),
    ]
    if args.default_prompt is not None:
        server_cmd += ["--default-prompt", args.default_prompt]

    server_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.server_mem_fraction),
    }

    output_display = (
        f"{args.intermediate_base_dir}/{args.run_dir_name}/job_results/{job_label}"
        if args.run_dir_name is not None
        else f"{args.output_dir}/{job_label}"
    )
    print("╔══ Classifier-guided eval job ══")
    print(f"║ Policy server : port {port}, {args.server_mem_fraction*100:.0f}% GPU mem")
    print(f"║ Evals         : {num_evals} envs, {eval_mem_each*100:.1f}% GPU mem each")
    print(f"║ Envs          : {args.envs}")
    print(f"║ Output        : {output_display}")
    if args.dry_run:
        print("║ ** DRY RUN **")
    print("╚══════════════════════════════")
    print()

    if args.dry_run:
        print("Policy server command (run from polaris root):")
        print(f"  cd {openpi_dir}")
        print(f"  CUDA_VISIBLE_DEVICES={args.gpu_id} XLA_PYTHON_CLIENT_MEM_FRACTION={args.server_mem_fraction} {' '.join(server_cmd)}")
        print()
        for i, job in enumerate(eval_jobs):
            cli = _eval_args_to_cli(job)
            env_str = f"POLARIS_CUDA_MEMORY_FRACTION={eval_mem_each} " if eval_mem_each > 0 else ""
            print(f"Eval [{i+1}] {job.environment}:")
            print(f"  {env_str}uv run python {eval_script} {' '.join(cli)}")
        return

    # Save job metadata when running in cluster mode (for postprocess)
    if args.run_dir_name is not None:
        job_output_root = Path(args.intermediate_base_dir) / args.run_dir_name / "job_results" / f"job_{args.job_id:03d}"
        job_output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_dir_name": args.run_dir_name,
            "job_id": args.job_id,
            "timestamp": timestamp,
            "actor_config": args.actor_config,
            "actor_dir": args.actor_dir,
            "classifier_config": args.classifier_config,
            "classifier_dir": args.classifier_dir,
            "num_candidates": args.num_candidates,
            "envs": args.envs,
            "rollouts": args.rollouts,
        }
        (job_output_root / "job_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Start policy server
    print("Starting policy server...")
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=openpi_dir,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        print(f"Waiting for server on {host}:{port} (timeout {args.server_ready_timeout}s)...")
        if not _wait_for_server(host, port, timeout_sec=args.server_ready_timeout):
            raise RuntimeError("Policy server did not become ready in time")

        print("Server ready. Launching eval processes...")

        eval_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
            "POLARIS_CUDA_MEMORY_FRACTION": str(eval_mem_each),
        }

        procs: list[tuple[str, subprocess.Popen]] = []
        for job in eval_jobs:
            cli = _eval_args_to_cli(job)
            cmd = ["uv", "run", "python", eval_script] + cli
            p = subprocess.Popen(
                cmd,
                env=eval_env,
            )
            procs.append((job.environment, p))

        # Wait for all evals
        failures: list[str] = []
        for env_name, p in procs:
            p.wait()
            if p.returncode != 0:
                failures.append(env_name)
                print(f"  [{env_name}] FAILED (exit {p.returncode})")
            else:
                print(f"  [{env_name}] DONE")

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    print()
    print(f"=== Job complete: {num_evals - len(failures)}/{num_evals} evals succeeded ===")
    if failures:
        print(f"    Failed: {failures}")


if __name__ == "__main__":
    main()
