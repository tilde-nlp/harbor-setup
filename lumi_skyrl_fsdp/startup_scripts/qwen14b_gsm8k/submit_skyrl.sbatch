#!/bin/bash
#SBATCH --job-name=skyrl-qwen14b-gsm8k
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g       
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export BINDS="/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038"
export SANDBOX="$ROOT/multitorch_sandbox.sif"
export SCRIPTS="$ROOT/scripts"
export GPUS_PER_NODE=8
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RAY_PORT=6379

export MODEL_SRC=Qwen/Qwen3-14B

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"

# The batch script runs on the first node of the allocation, which is also our Ray
# head node -- assert that so the local hsn0 lookup below can't silently be wrong.
[ "${SLURMD_NODENAME:-$(hostname)}" = "$HEAD_NODE" ] || {
  echo "FATAL: batch host ${SLURMD_NODENAME:-$(hostname)} != first node $HEAD_NODE" >&2
  exit 1
}

# Ray and RCCL traffic must stay on the Slingshot fabric (hsn*), not the management
# network. We're on the head node, so read its hsn0 address directly (no srun needed).
HEAD_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
echo "nodes: ${NODES[*]}   head: $HEAD_NODE ($RAY_ADDRESS)"

# Plain (non-overlap) per-node steps for the Ray nodes. Each step owns its whole
# node: 8 GCDs, 56 cores. ray_node.sh asserts it really got $GPUS_PER_NODE GPUs.
SRUN_RAY="srun --nodes=1 --ntasks=1 --gpus-per-node=$GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_TASK"

# 1. Ray head on this node
$SRUN_RAY -w "$HEAD_NODE" bash "$SCRIPTS/ray_node.sh" head &
sleep 20

# 2. Ray worker on every other node
for node in "${NODES[@]:1}"; do
  $SRUN_RAY -w "$node" bash "$SCRIPTS/ray_node.sh" worker &
done

# 3. Driver: runs here, inside the batch step (which holds all 8 GCDs and all
#    cores on this node -- no srun, no --overlap). The driver only talks to the
#    Ray head over TCP; GPU work happens in actors under the raylets, and those
#    inherit the raylet's environment, not the driver's.
#    run_gsm8k_2node.sh first waits until all GPUs have joined the cluster.
export BINDS="/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038"
singularity exec -B "$BINDS" "$SANDBOX" \
  bash -c "source $SCRIPTS/skyrl_env.sh && bash $SCRIPTS/run_training.sh"