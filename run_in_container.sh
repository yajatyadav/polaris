#!/bin/bash
# Wraps an existing shell script in an apptainer container with GPU support.
# Usage: ./run_in_container.sh <script.sh> [extra args...]

CONTAINER="${POLARIS_CONTAINER:-$(dirname "$0")/cuda-ubuntu22.sif}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script.sh> [extra args...]"
    exit 1
fi

SCRIPT="$(realpath "$1")"
shift

if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container not found at $CONTAINER"
    echo "Build it with: apptainer pull cuda-ubuntu22.sif docker://nvidia/cuda:12.3.0-devel-ubuntu22.04"
    exit 1
fi

apptainer exec --nv \
    --bind "$(dirname "$SCRIPT")" \
    "$CONTAINER" \
    bash "$SCRIPT" "$@"