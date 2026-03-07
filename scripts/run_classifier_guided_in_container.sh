#!/usr/bin/env bash
# Run classifier-guided job inside apptainer. Must be run from polaris root on a GPU node.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apptainer exec --nv --bind /usr/share/vulkan:/usr/share/vulkan \
  --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
  ./polaris.sif /bin/bash -c 'uv run python experiments/run_classifier_guided_job.py \
  --actor-config pi05_droid_jointpos_polaris \
  --actor-dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris \
  --classifier-config polaris_droid_language_classifier_joint_pos \
  --classifier-dir checkpoints/polaris_droid_language_classifier_joint_pos/polaris_droid_jointpos_lr5e-5_ema0.99_adamw_posloss0.95_b256/9999 \
  --num-candidates 1 \
  --envs DROID-BlockStackKitchen \
  --rollouts 1 \
  --gpu-id 0'
