#!/bin/bash
#SBATCH --job-name=rccl-test
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g
#SBATCH --nodes=4                     # same allocation shape as the SkyRL jobs
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

# Same cluster bring-up as submit_skyrl.sbatch (ray_symmetric.sh on every node),
# but the driver command is run_rccl_test.sh and no checkpoint is staged.

set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export BINDS="/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038"
export SANDBOX="$ROOT/multitorch_sandbox.sif"
export SCRIPTS="$ROOT/scripts"
export GPUS_PER_NODE=8
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RAY_PORT=6379

# What ray_symmetric.sh runs on the head node once the cluster is up.
export ENTRYPOINT="$SCRIPTS/run_rccl_test.sh"
# Test parameters, read by run_rccl_test.sh.
export NUM_RANKS="${NUM_RANKS:-3}"        # 3 actors, STRICT_SPREAD -> one rank per node, nNodes 3
export MASTER_PORT="${MASTER_PORT:-12355}"
# MODEL_SRC deliberately unset: ray_symmetric.sh skips the staging block.

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"
[ "${SLURMD_NODENAME:-$(hostname)}" = "$HEAD_NODE" ] || {
  echo "FATAL: batch host ${SLURMD_NODENAME:-$(hostname)} != first node $HEAD_NODE" >&2
  exit 1
}

HEAD_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
echo "nodes: ${NODES[*]}   head: $HEAD_NODE ($RAY_ADDRESS)   ranks: $NUM_RANKS"
echo "rccl logs: $ROOT/logs/nccl-$SLURM_JOB_ID"

srun --label --kill-on-bad-exit=1 \
     --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
     --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     bash "$SCRIPTS/ray_symmetric.sh"
