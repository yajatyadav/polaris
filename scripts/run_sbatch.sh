#!/usr/bin/env bash
# Wrapper for sbatch: runs the given command (e.g. uv run python ...).
# Usage: sbatch [options] scripts/run_sbatch.sh "uv run python ..."
set -euo pipefail
eval "$@"
