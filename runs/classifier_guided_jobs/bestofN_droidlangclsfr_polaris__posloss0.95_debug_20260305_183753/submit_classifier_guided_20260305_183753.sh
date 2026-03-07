#!/usr/bin/env bash
set -euo pipefail

# cd to polaris root (relative to this script: run_dir -> classifier_guided_jobs -> runs -> polaris)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

RUN_NAME="bestofN_droidlangclsfr_polaris__posloss0.95_debug"
RUN_ID="82047fe1960ad750"
RUN_DIR_NAME="bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753"
RUN_ROOT="runs/classifier_guided_jobs/bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753"

CONTAINER="/global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/polaris/polaris.sif"

LOG_DIR="runs/classifier_guided_jobs/bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753/slurm_logs"
mkdir -p "$LOG_DIR"

echo "Submitting classifier-guided eval jobs..."
# num_candidates_list=[1], num_evals_per_job=1
# 1 x 1 = 1 jobs
job_ids=()

jobid_0=$(XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 sbatch -A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n 1 -c 4 --qos=rail_gpu4_high -t 24:00:00 --mem=60G --parsable --requeue --job-name=clf_guided_eval_N1_C0 --comment='classifier_guided N=1 chunk=0' --output=runs/classifier_guided_jobs/bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753/slurm_logs/clf_guided_eval_N1_C0_%j.out --error=runs/classifier_guided_jobs/bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753/slurm_logs/clf_guided_eval_N1_C0_%j.err --wrap='apptainer exec --nv --bind /usr/share/vulkan:/usr/share/vulkan --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/polaris/polaris.sif /bin/bash -c '"'"'uv run python experiments/run_classifier_guided_job.py --actor-config pi05_droid_jointpos_polaris --actor-dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris --classifier-config polaris_droid_language_classifier_joint_pos --classifier-dir checkpoints/polaris_droid_language_classifier_joint_pos/polaris_droid_jointpos_lr5e-5_ema0.99_adamw_posloss0.95_b256/9999 --num-candidates 1 --run-dir-name bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753 --job-id 0 --intermediate-base-dir runs/classifier_guided_jobs --envs DROID-BlockStackKitchen --gpu-id 0 --server-mem-fraction 0.28 --eval-mem-total 0.65 --rollouts 1'"'"'')
echo "Submitted clf_guided_eval_N1_C0: $jobid_0"
job_ids+=("$jobid_0")

echo "Total jobs submitted: ${#job_ids[@]}"
echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run dir name: bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753"
echo "Intermediate outputs: $RUN_ROOT"

echo "After jobs finish, run postprocess:"
echo 'uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py --intermediate-run-dir runs/classifier_guided_jobs/bestofN_droidlangclsfr_polaris__posloss0.95_debug_20260305_183753'
