"""
Generate SLURM submission script for classifier-guided policy evaluation.

The policy server runs on a separate machine — this script generates eval jobs
that query it via --host/--port. Each job runs num_evals_per_job envs in parallel
for a specific num_candidates value.

Jobs are generated as a grid: (num_candidates, env_chunk). All 6 polaris envs are
split into chunks of size num_evals_per_job (e.g. 3 envs per job -> 2 chunks).
Total jobs = len(num_candidates_list) * ceil(6 / num_evals_per_job).

Usage:
    uv run python experiments/brc_eval/generate_classifier_guided_sbatch.py \\
        --run-name my_eval \\

        --host 10.0.0.1 --port 8000 \\
        --num-candidates-list 32 64 128 \\
        --num-evals-per-job 3 \\
        --rollouts 5
"""

from __future__ import annotations

import argparse
import math
import os
import secrets
import shlex
import time

ALL_POLARIS_ENVS = [
    "DROID-BlockStackKitchen",
    "DROID-FoodBussing",
    "DROID-PanClean",
    "DROID-MoveLatteCup",
    "DROID-OrganizeTools",
    "DROID-TapeIntoContainer",
]


def _q(x):
    return shlex.quote(str(x))


def generate_submission_script(args):
    run_id = args.run_id or secrets.token_hex(8)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{args.run_name}_{timestamp}"
    run_root = os.path.join(args.intermediate_base_dir, run_dir_name)
    os.makedirs(run_root, exist_ok=True)

    out_script = os.path.join(run_root, f"submit_classifier_guided_{timestamp}.sh")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append("# cd to polaris root (relative to this script: run_dir -> classifier_guided_jobs -> runs -> polaris)")
    lines.append('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"')
    lines.append('cd "$SCRIPT_DIR/../../.."')
    lines.append("")
    lines.append(f'RUN_NAME="{args.run_name}"')
    lines.append(f'RUN_ID="{run_id}"')
    lines.append(f'RUN_DIR_NAME="{run_dir_name}"')
    lines.append(f'RUN_ROOT="{run_root}"')
    lines.append("")

    # Container setup
    container_path = args.container
    if container_path:
        lines.append(f'CONTAINER="{container_path}"')
        lines.append("")

    if not args.run_local:
        log_dir = os.path.join(run_root, "slurm_logs")
        os.makedirs(log_dir, exist_ok=True)
        lines.append(f'LOG_DIR="{log_dir}"')
        lines.append('mkdir -p "$LOG_DIR"')
        lines.append("")

    sbatch_common = [
        f"-A {args.account}",
        f"-p {args.partition}",
        f"--gres=gpu:{args.gpu_type}:{args.num_gpus}",
        f"-N {args.num_nodes}",
        f"-n {args.num_tasks}",
        f"-c {args.cpus_per_task}",
        f"--qos={args.qos}",
        f"-t {args.time}",
        f"--mem={args.mem}",
        "--parsable",
    ]
    if args.requeue:
        sbatch_common.append("--requeue")
    sbatch_common_str = " ".join(sbatch_common)

    system_env_vars = [
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "OMP_NUM_THREADS=1",
    ]
    all_env_vars = " ".join(system_env_vars)

    worker_script_rel = "experiments/run_classifier_guided_job.py"

    envs_to_use = args.envs if args.envs is not None else ALL_POLARIS_ENVS
    num_chunks = math.ceil(len(envs_to_use) / args.num_evals_per_job)
    env_chunks = [
        envs_to_use[i : i + args.num_evals_per_job]
        for i in range(0, len(envs_to_use), args.num_evals_per_job)
    ]
    total_jobs = len(args.num_candidates_list) * num_chunks

    if args.run_local:
        lines.append('echo "Running classifier-guided eval jobs locally (sequential)..."')
    else:
        lines.append("echo \"Submitting classifier-guided eval jobs...\"")
    lines.append(f"# num_candidates_list={args.num_candidates_list}, num_evals_per_job={args.num_evals_per_job}")
    lines.append(f"# {len(args.num_candidates_list)} x {num_chunks} = {total_jobs} jobs")
    if not args.run_local:
        lines.append("job_ids=()")
    lines.append("")

    job_idx = 0
    for n_idx, num_candidates in enumerate(args.num_candidates_list):
        for chunk_id, env_chunk in enumerate(env_chunks):
            envs_str = " ".join(_q(e) for e in env_chunk)
            job_name = f"{args.slurm_job_name_prefix}_N{num_candidates}_C{chunk_id}"

            py_cmd_parts = [
                "uv run python",
                worker_script_rel,
                f"--host {_q(args.host)}",
                f"--port {args.port}",
                f"--num-candidates {num_candidates}",
                f"--run-dir-name {_q(run_dir_name)}",
                f"--job-id {job_idx}",
                f"--intermediate-base-dir {_q(args.intermediate_base_dir)}",
                f"--envs {envs_str}",
                f"--gpu-id {args.gpu_id}",
                f"--eval-mem-total {args.eval_mem_total}",
            ]
            py_cmd = " ".join(py_cmd_parts)
            if args.rollouts is not None:
                py_cmd += f" --rollouts {args.rollouts}"
            if args.default_prompt is not None:
                py_cmd += f" --default-prompt {_q(args.default_prompt)}"

            if args.run_local:
                if container_path:
                    run_line = (
                        f'{all_env_vars} apptainer exec --nv '
                        f'--bind /usr/share/vulkan:/usr/share/vulkan '
                        f'--env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json '
                        f'"$CONTAINER" '
                        f'/bin/bash -c {_q(py_cmd)}'
                    )
                else:
                    run_line = f"{all_env_vars} {args.script_runner} {_q(py_cmd)}"
                lines.append(f'echo "Running {job_name}..."')
                lines.append(run_line)
                lines.append("")
            else:
                stdout_path = os.path.join(log_dir, f"{job_name}_%j.out")
                stderr_path = os.path.join(log_dir, f"{job_name}_%j.err")
                if container_path:
                    wrapped_cmd = (
                        f'apptainer exec --nv '
                        f'--bind /usr/share/vulkan:/usr/share/vulkan '
                        f'--env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json '
                        f'{_q(container_path)} '
                        f'/bin/bash -c {_q(py_cmd)}'
                    )
                    sbatch_line = (
                        f'jobid_{job_idx}=$({all_env_vars} sbatch {sbatch_common_str} '
                        f'--job-name={_q(job_name)} '
                        f'--comment={_q(f"classifier_guided N={num_candidates} chunk={chunk_id}")} '
                        f'--output={_q(stdout_path)} '
                        f'--error={_q(stderr_path)} '
                        f'--wrap={_q(wrapped_cmd)})'
                    )
                else:
                    sbatch_line = (
                        f'jobid_{job_idx}=$({all_env_vars} sbatch {sbatch_common_str} '
                        f'--job-name={_q(job_name)} '
                        f'--comment={_q(f"classifier_guided N={num_candidates} chunk={chunk_id}")} '
                        f'--output={_q(stdout_path)} '
                        f'--error={_q(stderr_path)} '
                        f'{_q(args.script_runner)} {_q(py_cmd)})'
                    )
                lines.append(sbatch_line)
                lines.append(f'echo "Submitted {job_name}: $jobid_{job_idx}"')
                lines.append(f'job_ids+=("$jobid_{job_idx}")')
                lines.append("")
            job_idx += 1

    if args.run_local:
        lines.append('echo "All jobs finished."')
    else:
        lines.append('echo "Total jobs submitted: ${#job_ids[@]}"')
    lines.append('echo "Run name: $RUN_NAME"')
    lines.append('echo "Run ID: $RUN_ID"')
    lines.append(f'echo "Run dir name: {run_dir_name}"')
    lines.append('echo "Intermediate outputs: $RUN_ROOT"')
    lines.append("")
    lines.append("echo \"After jobs finish, run postprocess:\"")
    lines.append(
        "echo "
        + _q(
            "uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py "
            f"--intermediate-run-dir {run_root}"
        )
    )

    with open(out_script, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(out_script, 0o755)

    print(f"Generated: {out_script}")
    print(f"Run ID: {run_id}")
    print(f"Intermediate run directory: {run_root}")
    print(f"Submit with: bash {out_script}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SLURM script for classifier-guided eval.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--run-id", type=str, default="")

    # Remote policy server
    parser.add_argument("--host", type=str, required=True, help="Policy server host")
    parser.add_argument("--port", type=int, required=True, help="Policy server port")

    parser.add_argument(
        "--num-candidates-list",
        nargs="+",
        type=int,
        required=True,
        help="List of num_candidates values to sweep (e.g. 32 64 128)",
    )
    parser.add_argument("--default-prompt", type=str, default=None)

    parser.add_argument(
        "--num-evals-per-job",
        type=int,
        default=3,
        help="Envs per job (6 total, so 3 -> 2 jobs per num_candidates). Use 1 for debug.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        type=str,
        default=None,
        help="Override envs to evaluate (default: all 6).",
    )
    parser.add_argument("--rollouts", type=int, default=None)

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-mem-total", type=float, default=0.65)

    parser.add_argument("--intermediate-base-dir", type=str, default="runs/classifier_guided_jobs")
    parser.add_argument(
        "--run-local",
        action="store_true",
        default=False,
        help="Generate script for local run (no sbatch). Jobs run sequentially.",
    )
    parser.add_argument(
        "--script-runner",
        type=str,
        default="scripts/run_sbatch.sh",
        help="Wrapper script for sbatch (receives py_cmd as arg). Use 'bash -c' for inline.",
    )
    parser.add_argument(
        "--container",
        type=str,
        default='/global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/polaris/polaris.sif',
        help="Path to Apptainer/Singularity .sif container.",
    )

    parser.add_argument("--account", type=str, default="co_rail")
    parser.add_argument("--partition", type=str, default="savio4_gpu")
    parser.add_argument("--gpu-type", type=str, default="A5000")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--qos", type=str, default="rail_gpu4_high")
    parser.add_argument("--time", type=str, default="24:00:00")
    parser.add_argument("--mem", type=str, default="60G")
    parser.add_argument("--requeue", action="store_true", default=True)
    parser.add_argument("--no-requeue", action="store_false", dest="requeue")
    parser.add_argument("--slurm-job-name-prefix", type=str, default="clf_guided_eval")
    return parser.parse_args()


if __name__ == "__main__":
    generate_submission_script(parse_args())
