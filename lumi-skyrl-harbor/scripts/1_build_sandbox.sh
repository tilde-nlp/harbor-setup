#!/bin/bash
set -euo pipefail

ROOT=/scratch/project_465003062/environment/containers/skyrl-harbor
SIF="$ROOT/lumi-multitorch-plus.sif"

export SINGULARITY_TMPDIR="$ROOT/.singtmp"
export SINGULARITY_CACHEDIR="$ROOT/.singcache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

srun --account=project_465003062 \
     --partition=small --cpus-per-task=56 --mem=64G --time=01:00:00 \
  singularity build --sandbox "$ROOT/multitorch_sandbox" "$SIF"