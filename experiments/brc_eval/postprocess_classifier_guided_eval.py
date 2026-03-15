"""
Aggregate classifier-guided eval job results and write final outputs.

Run after all SLURM jobs finish. Loads CSVs from job_results, aggregates
success/progress per (num_candidates, env), and writes a summary.

Usage:
    uv run python experiments/brc_eval/postprocess_classifier_guided_eval.py \
        --intermediate-run-dir runs/classifier_guided_jobs/my_run_20260304_123456
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd


def _load_job_results(intermediate_run_dir: str) -> dict[tuple[int, str], pd.DataFrame]:
    """Load all eval_results.csv from job_results/job_*/N*/*/."""
    job_results_dir = Path(intermediate_run_dir) / "job_results"
    if not job_results_dir.exists():
        raise FileNotFoundError(f"Job results dir not found: {job_results_dir}")

    # Try new structure: job_*/N*/*/eval_results.csv
    pattern_new = str(job_results_dir / "job_*" / "N*" / "*" / "eval_results.csv")
    # Fallback: old structure job_*/*/eval_results.csv
    pattern_old = str(job_results_dir / "job_*" / "*" / "eval_results.csv")

    csv_paths = sorted(glob.glob(pattern_new))
    use_new_structure = len(csv_paths) > 0
    if not use_new_structure:
        csv_paths = sorted(glob.glob(pattern_old))
    if not csv_paths:
        raise FileNotFoundError(f"No eval_results.csv found under: {job_results_dir}")

    # Group by (num_candidates, env_name)
    candidates: dict[tuple[int, str], list[tuple[pd.DataFrame, Path]]] = {}
    for p in csv_paths:
        path = Path(p)
        if use_new_structure:
            # path: .../job_000/N64/DROID-FoodBussing_20260304_123456/eval_results.csv
            n_part = path.parent.parent.name  # N64
            num_cand = int(n_part[1:])  # strip "N"
        else:
            # Try to get from job_metadata.json
            job_dir = path.parent.parent
            meta_path = job_dir / "job_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                num_cand = meta.get("num_candidates", 1)
            else:
                num_cand = 1

        env_part = path.parent.name  # DROID-FoodBussing_20260304_123456
        parts = env_part.rsplit("_", 2)
        env_name = parts[0] if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit() else env_part

        key = (num_cand, env_name)
        df = pd.read_csv(p)
        candidates.setdefault(key, []).append((df, path))

    results: dict[tuple[int, str], pd.DataFrame] = {}
    for key, pairs in candidates.items():
        best_df, _ = max(pairs, key=lambda x: (len(x[0]), x[1].stat().st_mtime))
        results[key] = best_df

    return results


def _aggregate_results(results: dict[tuple[int, str], pd.DataFrame]) -> dict:
    """Aggregate per-(num_candidates, env) and overall metrics."""
    per_n: dict[int, dict[str, dict]] = {}
    all_episodes: list[dict] = []

    for (num_cand, env_name), df in results.items():
        if df.empty:
            continue
        per_n.setdefault(num_cand, {})
        success_mean = df["success"].mean()
        progress_mean = df["progress"].mean()
        n_episodes = len(df)
        per_n[num_cand][env_name] = {
            "success": float(success_mean),
            "progress": float(progress_mean),
            "n_episodes": n_episodes,
        }
        for _, row in df.iterrows():
            all_episodes.append({
                "num_candidates": num_cand,
                "env": env_name,
                "episode": int(row["episode"]),
                "success": bool(row["success"]),
                "progress": float(row["progress"]),
                "episode_length": int(row["episode_length"]),
            })

    # Per-N overall
    per_num_candidates = {}
    for num_cand, envs in sorted(per_n.items()):
        total_ep = sum(e["n_episodes"] for e in envs.values())
        overall_success = sum(e["success"] * e["n_episodes"] for e in envs.values()) / max(total_ep, 1)
        overall_progress = sum(e["progress"] * e["n_episodes"] for e in envs.values()) / max(total_ep, 1)
        per_num_candidates[num_cand] = {
            "overall_success": float(overall_success),
            "overall_progress": float(overall_progress),
            "total_episodes": total_ep,
            "per_env": envs,
        }

    # Grand overall
    grand_total = sum(v["total_episodes"] for v in per_num_candidates.values())
    grand_success = sum(v["overall_success"] * v["total_episodes"] for v in per_num_candidates.values()) / max(grand_total, 1)
    grand_progress = sum(v["overall_progress"] * v["total_episodes"] for v in per_num_candidates.values()) / max(grand_total, 1)

    return {
        "per_num_candidates": per_num_candidates,
        "overall": {
            "success": float(grand_success),
            "progress": float(grand_progress),
            "total_episodes": grand_total,
        },
        "all_episodes": all_episodes,
    }


def _write_final_outputs(args, aggregated: dict, metadata: dict | None) -> None:
    save_dir = Path(args.results_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.run_name

    summary = {
        "run_name": args.run_name,
        "run_id": args.run_id,
        "intermediate_run_dir": args.intermediate_run_dir,
        "overall": aggregated["overall"],
        "per_num_candidates": {str(k): v for k, v in aggregated["per_num_candidates"].items()},
        "metadata": metadata,
    }
    summary_path = save_dir / f"{run_label}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")

    # Per-env CSV with num_candidates column
    per_env_path = save_dir / f"{run_label}_per_env.csv"
    rows = []
    for num_cand, data in sorted(aggregated["per_num_candidates"].items()):
        for env, metrics in data["per_env"].items():
            rows.append({"num_candidates": num_cand, "env": env, **metrics})
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
    for num_cand, data in aggregated["per_num_candidates"].items():
        log_data[f"eval_N{num_cand}/overall_success"] = data["overall_success"]
        log_data[f"eval_N{num_cand}/overall_progress"] = data["overall_progress"]
        for env, metrics in data["per_env"].items():
            safe_env = "".join(c if c.isalnum() or c in "_-" else "_" for c in env)
            log_data[f"eval_N{num_cand}/{safe_env}/success"] = metrics["success"]
            log_data[f"eval_N{num_cand}/{safe_env}/progress"] = metrics["progress"]
    run.log(log_data)
    print(f"Logged to wandb: {run.url}")
    run.finish()


def main(args) -> None:
    if not args.results_save_path:
        args.results_save_path = args.intermediate_run_dir
    if not args.run_name:
        args.run_name = os.path.basename(os.path.normpath(args.intermediate_run_dir))

    results = _load_job_results(args.intermediate_run_dir)
    print(f"Loaded results for {len(results)} (num_candidates, env) combinations.")

    # Load metadata from first job if available
    job_results_dir = Path(args.intermediate_run_dir) / "job_results"
    metadata = None
    for meta_path in sorted(job_results_dir.glob("job_*/N*/job_metadata.json")):
        with open(meta_path) as f:
            metadata = json.load(f)
        break
    if metadata is None:
        for meta_path in sorted(job_results_dir.glob("job_*/job_metadata.json")):
            with open(meta_path) as f:
                metadata = json.load(f)
            break

    aggregated = _aggregate_results(results)
    print(f"Aggregated: {len(aggregated['per_num_candidates'])} num_candidates values, {aggregated['overall']['total_episodes']} total episodes")
    for num_cand, data in sorted(aggregated["per_num_candidates"].items()):
        print(f"  N={num_cand}: success={data['overall_success']:.2%}, progress={data['overall_progress']:.2%}, episodes={data['total_episodes']}")

    _write_final_outputs(args, aggregated, metadata)
    if args.use_wandb:
        _log_to_wandb(args, aggregated)


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess classifier-guided eval results.")
    parser.add_argument("--intermediate-run-dir", type=str, required=True)
    parser.add_argument("--results-save-path", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")

    parser.add_argument("--wandb-project", type=str, default="polaris-evals")
    parser.add_argument("--wandb-entity", type=str, default="yajatyadav")
    parser.add_argument("--wandb-group-name", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
