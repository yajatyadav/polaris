# Classifier-guided policy evaluation (cluster workflow)

This folder contains a cluster workflow that mirrors `experiments/run_classifier_guided_job.py` but splits work into SLURM jobs with unique logging and a postprocess step:

- one job = one policy server + N parallel env evals on one GPU
- per-job outputs go to `runs/classifier_guided_jobs/{run_name}_{timestamp}`
- postprocess writes summary + per_env.csv into the same run dir

## Files

- `generate_classifier_guided_sbatch.py`: creates a submission script with all `sbatch` calls and unique log paths
- `run_classifier_guided_job.py`: worker (in `experiments/`) — one policy server + N evals
- `postprocess_classifier_guided_eval.py`: aggregate all job outputs, write summary + per_env.csv into run dir

## Typical workflow

1. Generate submission script (add `--run-local` for local machine, no sbatch):

```bash
uv run python experiments/brc_eval/generate_classifier_guided_sbatch.py \
  --run-name my_classifier_eval \
  --actor-config pi05_droid_jointpos_polaris \
  --actor-dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris \
  --classifier-config polaris_droid_language_classifier_joint_pos \
  --classifier-dir checkpoints/polaris_droid_language_classifier_joint_pos/.../2000 \
  --num-candidates-list 32 64 \
  --num-evals-per-job 3 \
  --rollouts 5
```

2. Submit jobs:

```bash
bash runs/classifier_guided_jobs/{run_name}_{timestamp}/submit_classifier_guided_{timestamp}.sh
```

3. After all jobs finish, run postprocess (just the run dir):

```bash
uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py \
  --intermediate-run-dir runs/classifier_guided_jobs/{run_name}_{timestamp}
```

## Unique logging

- Each run gets `{run_name}_{timestamp}` so runs never collide
- Each job writes to `job_results/job_{id}/` with unique per-env folders `{env}_{timestamp}`
- SLURM logs go to `slurm_logs/{job_name}_{%j}.out/err`

## Local run (no SLURM)

**Option A: Same script as cluster** — generate with `--run-local` to get a script that runs jobs sequentially (no sbatch):

```bash
uv run python experiments/brc_eval/generate_classifier_guided_sbatch.py \
  --run-name my_local_run --run-local \
  --actor-config ... --actor-dir ... \
  --classifier-config ... --classifier-dir ... \
  --num-candidates-list 32 --num-evals-per-job 3 --rollouts 5
bash runs/classifier_guided_jobs/my_local_run_*/submit_classifier_guided_*.sh
```

**Option B: Single job** — run one job directly:

```bash
uv run python experiments/run_classifier_guided_job.py \
  --actor-config ... --actor-dir ... \
  --classifier-config ... --classifier-dir ... \
  --envs DROID-FoodBussing DROID-PanClean \
  --rollouts 5
```

For cluster-style output paths, add `--run-dir-name my_run_20260304_123456 --job-id 0 --intermediate-base-dir runs/classifier_guided_jobs`.
