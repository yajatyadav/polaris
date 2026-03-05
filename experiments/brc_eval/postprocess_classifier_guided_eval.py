"""
Aggregate classifier-guided eval job results and write final outputs.

Run after all SLURM jobs finish. Loads CSVs from job_results, aggregates
success/progress per env, and writes a summary. Optionally logs to wandb.

Usage:
    # Minimal: just the run dir (writes summary + per_env.csv there)
    uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py \\
        --intermediate-run-dir runs/classifier_guided_jobs/my_run_20260304_123456

    # Optional: override save location, run-name, run-id for wandb
    uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py \\
        --intermediate-run-dir runs/classifier_guided_jobs/my_run_20260304_123456 \\
        --run-name my_run --run-id abc123 --use-wandb
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd


def _load_job_results(intermediate_run_dir: str) -> tuple[dict, list[Path]]:
    """Load all eval_results.csv from job_results/job_*/*/."""
    job_results_dir = Path(intermediate_run_dir) / "job_results"
    if not job_results_dir.exists():
        raise FileNotFoundError(f"Job results dir not found: {job_results_dir}")

    pattern = str(job_results_dir / "job_*" / "*" / "eval_results.csv")
    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No eval_results.csv found under: {job_results_dir}")

    # Group by (job_id, env_name) - take most complete if multiple (e.g. retries)
    candidates: dict[tuple[int, str], list[tuple[pd.DataFrame, Path]]] = {}
    for p in csv_paths:
        path = Path(p)
        job_part = path.parent.parent.name  # job_000
        env_part = path.parent.name  # DROID-FoodBussing_20260304_123456
        parts = env_part.rsplit("_", 2)
        env_name = parts[0] if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit() else env_part
        job_id = int(job_part.replace("job_", ""))
        key = (job_id, env_name)
        df = pd.read_csv(p)
        candidates.setdefault(key, []).append((df, path))

    results: dict[tuple[int, str], pd.DataFrame] = {}
    for key, pairs in candidates.items():
        # Prefer most rows, then latest mtime
        best_df, _ = max(pairs, key=lambda x: (len(x[0]), x[1].stat().st_mtime))
        results[key] = best_df

    return results, [Path(p) for p in csv_paths]


def _aggregate_results(results: dict[tuple[int, str], pd.DataFrame]) -> dict:
    """Aggregate per-env and overall metrics."""
    per_env: dict[str, dict] = {}
    all_episodes: list[dict] = []

    for (job_id, env_name), df in results.items():
        if df.empty:
            continue
        success_mean = df["success"].mean()
        progress_mean = df["progress"].mean()
        n_episodes = len(df)
        per_env[env_name] = {
            "success": float(success_mean),
            "progress": float(progress_mean),
            "n_episodes": n_episodes,
        }
        for _, row in df.iterrows():
            all_episodes.append({
                "job_id": job_id,
                "env": env_name,
                "episode": int(row["episode"]),
                "success": bool(row["success"]),
                "progress": float(row["progress"]),
                "episode_length": int(row["episode_length"]),
            })

    overall_success = sum(e["success"] * e["n_episodes"] for e in per_env.values()) / max(
        sum(e["n_episodes"] for e in per_env.values()), 1
    )
    overall_progress = sum(e["progress"] * e["n_episodes"] for e in per_env.values()) / max(
        sum(e["n_episodes"] for e in per_env.values()), 1
    )

    return {
        "per_env": per_env,
        "overall": {
            "success": float(overall_success),
            "progress": float(overall_progress),
            "total_episodes": sum(e["n_episodes"] for e in per_env.values()),
        },
        "all_episodes": all_episodes,
    }


def _write_final_outputs(args, aggregated: dict, metadata: dict | None) -> None:
    # Default: write directly into intermediate_run_dir
    save_dir = Path(args.results_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.run_name  # e.g. "my_run_20260304_123456"

    summary = {
        "run_name": args.run_name,
        "run_id": args.run_id,
        "intermediate_run_dir": args.intermediate_run_dir,
        "overall": aggregated["overall"],
        "per_env": aggregated["per_env"],
        "metadata": metadata,
    }
    summary_path = save_dir / f"{run_label}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")

    # Also save per-env CSV for easy inspection
    per_env_path = save_dir / f"{run_label}_per_env.csv"
    rows = [
        {"env": env, **data}
        for env, data in aggregated["per_env"].items()
    ]
    pd.DataFrame(rows).to_csv(per_env_path, index=False)
    print(f"Wrote per-env: {per_env_path}")


def _log_to_wandb(args, aggregated: dict) -> None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping wandb logging")
        return

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group_name or args.run_name,
        name=args.wandb_run_name or args.run_name,
        id=args.run_id if args.run_id else None,
        resume="allow",
    )
    log_data = {
        "eval/overall_success": aggregated["overall"]["success"],
        "eval/overall_progress": aggregated["overall"]["progress"],
        "eval/total_episodes": aggregated["overall"]["total_episodes"],
    }
    for env, data in aggregated["per_env"].items():
        safe_env = "".join(c if c.isalnum() or c in "_-" else "_" for c in env)
        log_data[f"eval_env/{safe_env}/success"] = data["success"]
        log_data[f"eval_env/{safe_env}/progress"] = data["progress"]
        log_data[f"eval_env/{safe_env}/n_episodes"] = data["n_episodes"]
    run.log(log_data)
    print(f"Logged to wandb: {run.url}")
    run.finish()


def main(args) -> None:
    # Default results_save_path = intermediate_run_dir (write CSVs in same place)
    if not args.results_save_path:
        args.results_save_path = args.intermediate_run_dir
    # Default run_name = basename of intermediate_run_dir
    if not args.run_name:
        args.run_name = os.path.basename(os.path.normpath(args.intermediate_run_dir))

    results, paths = _load_job_results(args.intermediate_run_dir)
    print(f"Loaded {len(paths)} eval result files.")

    # Load metadata from first job if available
    job_results_dir = Path(args.intermediate_run_dir) / "job_results"
    metadata = None
    for meta_path in sorted(job_results_dir.glob("job_*/job_metadata.json")):
        with open(meta_path) as f:
            metadata = json.load(f)
        break

    aggregated = _aggregate_results(results)
    print(f"Aggregated: {len(aggregated['per_env'])} envs, {aggregated['overall']['total_episodes']} episodes")
    print(f"  Overall success: {aggregated['overall']['success']:.2%}")
    print(f"  Overall progress: {aggregated['overall']['progress']:.2%}")

    _write_final_outputs(args, aggregated, metadata)
    if args.use_wandb:
        _log_to_wandb(args, aggregated)


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess classifier-guided eval results.")
    parser.add_argument("--intermediate-run-dir", type=str, required=True)
    parser.add_argument(
        "--results-save-path",
        type=str,
        default="",
        help="Where to write summary + per_env.csv. Default: same as intermediate-run-dir.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run label for output filenames. Default: basename of intermediate-run-dir.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Optional, for wandb resume.")

    parser.add_argument("--wandb-project", type=str, default="polaris")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group-name", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
