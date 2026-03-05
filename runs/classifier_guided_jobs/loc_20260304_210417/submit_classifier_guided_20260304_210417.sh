#!/usr/bin/env bash
set -euo pipefail

# cd to polaris root (relative to this script: run_dir -> classifier_guided_jobs -> runs -> polaris)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

RUN_NAME="loc"
RUN_ID="79d8c006a0ff8e44"
RUN_DIR_NAME="loc_20260304_210417"
RUN_ROOT="runs/classifier_guided_jobs/loc_20260304_210417"

echo "Running classifier-guided eval jobs locally (sequential)..."
# num_candidates_list=[32], num_evals_per_job=3
# 1 x 2 = 2 jobs

echo "Running clf_guided_eval_N32_C0..."
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 scripts/run_sbatch.sh 'uv run python experiments/run_classifier_guided_job.py --actor-config x --actor-dir y --classifier-config z --classifier-dir w --num-candidates 32 --run-dir-name loc_20260304_210417 --job-id 0 --intermediate-base-dir runs/classifier_guided_jobs --envs DROID-BlockStackKitchen DROID-FoodBussing DROID-PanClean --gpu-id 0 --server-mem-fraction 0.35 --eval-mem-total 0.6'

echo "Running clf_guided_eval_N32_C1..."
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 scripts/run_sbatch.sh 'uv run python experiments/run_classifier_guided_job.py --actor-config x --actor-dir y --classifier-config z --classifier-dir w --num-candidates 32 --run-dir-name loc_20260304_210417 --job-id 1 --intermediate-base-dir runs/classifier_guided_jobs --envs DROID-MoveLatteCup DROID-OrganizeTools DROID-TapeIntoContainer --gpu-id 0 --server-mem-fraction 0.35 --eval-mem-total 0.6'

echo "All jobs finished."
echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run dir name: loc_20260304_210417"
echo "Intermediate outputs: $RUN_ROOT"

echo "After jobs finish, run postprocess:"
echo 'uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py --intermediate-run-dir runs/classifier_guided_jobs/loc_20260304_210417'
