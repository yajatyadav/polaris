#!/usr/bin/env bash
set -euo pipefail

# cd to polaris root (relative to this script: submission_scripts -> run_dir -> classifier_guided_jobs -> runs -> polaris)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."

RUN_NAME="dummy_run"
RUN_ID="f7cea8e638b530c5"
RUN_DIR_NAME="dummy_run_20260304_205606"
RUN_ROOT="runs/classifier_guided_jobs/dummy_run_20260304_205606"
LOG_DIR="runs/classifier_guided_jobs/dummy_run_20260304_205606/slurm_logs"

mkdir -p "$LOG_DIR"

echo "Submitting classifier-guided eval jobs..."
# num_candidates_list=[32], num_evals_per_job=3
# 1 x 2 = 2 jobs
job_ids=()

jobid_0=$(XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 sbatch -A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n 1 -c 4 --qos=rail_gpu4_high -t 24:00:00 --mem=60G --parsable --requeue --job-name=clf_guided_eval_N32_C0 --comment='classifier_guided N=32 chunk=0' --output=runs/classifier_guided_jobs/dummy_run_20260304_205606/slurm_logs/clf_guided_eval_N32_C0_%j.out --error=runs/classifier_guided_jobs/dummy_run_20260304_205606/slurm_logs/clf_guided_eval_N32_C0_%j.err scripts/run_sbatch.sh 'uv run python experiments/run_classifier_guided_job.py --actor-config dummy_actor --actor-dir /tmp/dummy --classifier-config dummy_clf --classifier-dir /tmp/dummy --num-candidates 32 --run-dir-name dummy_run_20260304_205606 --job-id 0 --intermediate-base-dir runs/classifier_guided_jobs --envs DROID-BlockStackKitchen DROID-FoodBussing DROID-PanClean --gpu-id 0 --server-mem-fraction 0.35 --eval-mem-total 0.6')
echo "Submitted clf_guided_eval_N32_C0: $jobid_0"
job_ids+=("$jobid_0")

jobid_1=$(XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 sbatch -A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n 1 -c 4 --qos=rail_gpu4_high -t 24:00:00 --mem=60G --parsable --requeue --job-name=clf_guided_eval_N32_C1 --comment='classifier_guided N=32 chunk=1' --output=runs/classifier_guided_jobs/dummy_run_20260304_205606/slurm_logs/clf_guided_eval_N32_C1_%j.out --error=runs/classifier_guided_jobs/dummy_run_20260304_205606/slurm_logs/clf_guided_eval_N32_C1_%j.err scripts/run_sbatch.sh 'uv run python experiments/run_classifier_guided_job.py --actor-config dummy_actor --actor-dir /tmp/dummy --classifier-config dummy_clf --classifier-dir /tmp/dummy --num-candidates 32 --run-dir-name dummy_run_20260304_205606 --job-id 1 --intermediate-base-dir runs/classifier_guided_jobs --envs DROID-MoveLatteCup DROID-OrganizeTools DROID-TapeIntoContainer --gpu-id 0 --server-mem-fraction 0.35 --eval-mem-total 0.6')
echo "Submitted clf_guided_eval_N32_C1: $jobid_1"
job_ids+=("$jobid_1")

echo "Total jobs submitted: ${#job_ids[@]}"
echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run dir name: dummy_run_20260304_205606"
echo "Intermediate outputs: $RUN_ROOT"

echo "After jobs finish, run postprocess:"
echo 'uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py --intermediate-run-dir runs/classifier_guided_jobs/dummy_run_20260304_205606 --results-save-path runs/classifier_guided_results --run-name dummy_run --run-id f7cea8e638b530c5'
