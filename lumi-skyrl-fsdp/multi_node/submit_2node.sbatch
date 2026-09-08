#!/bin/bash
#SBATCH --job-name=skyrl-gsm8k-14b
#SBATCH --account=project_465003062
#SBATCH --partition=standard-g        # dev-g works too if its node/time limits allow 2 nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0                       # whole node: colocated training offloads policy state to host RAM
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Multi-node SkyRL on LUMI-G.
#   sbatch $ROOT/scripts/submit_2node.sbatch
#
# Layout (all on shared scratch, visible from every node and bound into the sandbox):
#   $ROOT/scripts/submit_2node.sbatch   this file (runs on the host side, on the head node)
#   $ROOT/scripts/ray_node.sh           per-node helper: enters the container, starts one Ray node
#   $ROOT/scripts/skyrl_env.sh          env sourced INSIDE the container by every Ray node and the driver
#   $ROOT/scripts/run_gsm8k_2node.sh    the training script (runs inside the container on the head node)
#
# Change --nodes to 4 for Qwen3-32B; nothing else here needs editing, the training script
# derives NUM_GPUS from SLURM_JOB_NUM_NODES * GPUS_PER_NODE.
#
# NOTE on step layout: no srun step here uses --overlap, deliberately. Slurm 25.05
# (on LUMI since the 2026-09-02 maintenance) allocates only 7 of 8 GCDs to any
# --overlap step that covers a subset of a multi-node job, and blocks the 8th in
# the step's device cgroup (fixed upstream only in 26.05.3). Plain steps get all 8.
# So the ray nodes run as plain steps (one per node, never sharing a node), and the
# driver runs directly in this batch step instead of its own srun step.

set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export SANDBOX="$ROOT/multitorch_sandbox"
export SCRIPTS="$ROOT/scripts"
export GPUS_PER_NODE=8
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RAY_PORT=6379

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
export BINDS="/var/spool/slurmd,$ROOT,/scratch/project_465002038"
singularity exec -B "$BINDS" "$SANDBOX" \
  bash -c "source $SCRIPTS/skyrl_env.sh && bash $SCRIPTS/run_gsm8k_2node.sh"

# When the driver returns, this script exits and Slurm tears down the ray --block
# steps on all nodes, so nothing survives into the next job.
