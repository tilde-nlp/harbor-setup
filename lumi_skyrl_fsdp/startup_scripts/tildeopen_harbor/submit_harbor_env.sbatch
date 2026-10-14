#!/bin/bash
#SBATCH --job-name=harbor-env
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g           # smoke: 1 shared node. Scale: --partition=standard, whole nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/logs/harbor-env-%j.out
set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export SCRIPTS=$ROOT/scripts
export WORKERS_PER_NODE=${WORKERS_PER_NODE:-8}   # total workers >= SkyRL's max_concurrency
export BRIDGE_PORT=${BRIDGE_PORT:-9910}
BRIDGE_FILE=$ROOT/bridge.url
mkdir -p $ROOT/sif_cache $ROOT/singularity_cache $ROOT/logs

# 1. bridge, host side of node 0 (this batch host). Advertise the Slingshot address.
BRIDGE_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
export BRIDGE_URL="http://$BRIDGE_IP:$BRIDGE_PORT"
( module load cray-python && exec python3 \
    $ROOT/marin-harbor/src/harbor/environments/apptainer/server.py --port $BRIDGE_PORT ) \
    > $ROOT/logs/bridge-$SLURM_JOB_ID.log 2>&1 &
sleep 3
curl -sf "$BRIDGE_URL/status" >/dev/null || { echo "FATAL: bridge failed to start" >&2; exit 1; }

# 2. publish the address; withdraw it whenever this job ends for any reason.
echo "$BRIDGE_URL" > "$BRIDGE_FILE"
trap 'rm -f "$BRIDGE_FILE"' EXIT
echo "bridge $BRIDGE_URL (job $SLURM_JOB_ID) published to $BRIDGE_FILE"

# 3. one worker process per node (node 0 included), blocks for the job lifetime.
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK --export=ALL \
     bash $SCRIPTS/harbor_worker_node.sh
