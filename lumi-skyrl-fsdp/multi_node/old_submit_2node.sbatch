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
#   $ROOT/scripts/submit_2node.sbatch   this file (runs on the host side)
#   $ROOT/scripts/ray_node.sh           per-node helper: enters the container, starts one Ray node
#   $ROOT/scripts/skyrl_env.sh          env sourced INSIDE the container by every Ray node and the driver
#   $ROOT/scripts/run_gsm8k_2node.sh    the training script (runs inside the container on the head node)
#
# Change --nodes to 4 for Qwen3-32B; nothing else here needs editing, the training script
# derives NUM_GPUS from SLURM_JOB_NUM_NODES * GPUS_PER_NODE.

set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export SANDBOX="$ROOT/multitorch_sandbox"
export SCRIPTS="$ROOT/scripts"
export GPUS_PER_NODE=8
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RAY_PORT=6379

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"

# Ray and RCCL traffic must stay on the Slingshot fabric (hsn*), not the management network,
# so resolve the head's hsn0 address explicitly instead of letting Ray guess.
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
          bash -c "ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1")
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
echo "nodes: ${NODES[*]}   head: $HEAD_NODE ($RAY_ADDRESS)"

# Slurm 25.05 (LUMI since 2026-09-02) drops one GPU from any --overlap step
# that covers a subset of a multi-node job (fixed upstream in 26.05.3, not
# backported to 25.05). Plain steps are unaffected, so: no --overlap on the
# GPU (ray) steps; only the driver keeps it, since it shares the head node
# with the ray-head step (an --overlap step shares with non-overlap steps).
SRUN_RAY="srun --nodes=1 --ntasks=1 --gpus-per-node=$GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_TASK"
SRUN_DRV="srun --nodes=1 --ntasks=1 --overlap --cpus-per-task=$SLURM_CPUS_PER_TASK"

# 1. Ray head on node 0
$SRUN_RAY -w "$HEAD_NODE" bash "$SCRIPTS/ray_node.sh" head &
sleep 20

# 2. Ray worker on every other node
for node in "${NODES[@]:1}"; do
  $SRUN_RAY -w "$node" bash "$SCRIPTS/ray_node.sh" worker &
done

# 3. Driver on the head node. run_gsm8k_2node.sh first waits until all GPUs have joined.
$SRUN_DRV -w "$HEAD_NODE" \
  singularity exec -B /var/spool/slurmd,"$ROOT" "$SANDBOX" \
    bash -c "source $SCRIPTS/skyrl_env.sh && bash $SCRIPTS/run_gsm8k_2node.sh"

# When the driver step returns the sbatch script exits and Slurm tears down the
# ray --block steps on all nodes, so nothing survives into the next job.
