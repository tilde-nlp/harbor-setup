#!/bin/bash
# Runs host-side on one node of the harbor env job (one srun task per node).
# Starts the worker.py that owns `singularity instance start` for this node.
# Expects ROOT, BRIDGE_URL, WORKERS_PER_NODE from the sbatch env.
set -euo pipefail
module load cray-python                       # host python3 is 3.6; worker needs >=3.7
APPT="$ROOT/marin-harbor/src/harbor/environments/apptainer"
export SINGULARITY_CACHEDIR="$ROOT/singularity_cache"
echo "[$SLURMD_NODENAME] worker x$WORKERS_PER_NODE -> $BRIDGE_URL"
exec python3 "$APPT/worker.py" --bridge-url "$BRIDGE_URL" \
     --sif-cache "$ROOT/sif_cache" --num-workers "$WORKERS_PER_NODE"