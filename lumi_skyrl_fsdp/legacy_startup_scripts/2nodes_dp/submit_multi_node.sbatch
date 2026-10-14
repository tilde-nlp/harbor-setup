#!/bin/bash
#SBATCH --job-name=skyrl-gsm8k-14b
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0                       # whole node
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Multi-node SkyRL on LUMI-G, Ray cluster started with `ray symmetric-run`.
#   sbatch $ROOT/scripts/submit_2node.sbatch
#
# Layout (all on shared scratch, visible from every node and bound into the sandbox):
#   $ROOT/scripts/submit_2node.sbatch   this file (host side, runs on the first node)
#   $ROOT/scripts/ray_symmetric.sh      per-node wrapper: enters the container, runs ray symmetric-run
#   $ROOT/scripts/skyrl_env.sh          env sourced INSIDE the container on every node
#   $ROOT/scripts/run_gsm8k_2node.sh    training script, run by symmetric-run on the head node only
#
# How it works: one plain srun step, one task per node. Every task runs
# `ray symmetric-run --address <head hsn0 ip>:6379`. The task whose interfaces
# include that address becomes the head (starts `ray start --head`, waits for
# --min-nodes, runs the training script, then `ray stop`); every other task
# starts a worker raylet and blocks. Exit code of the training script becomes
# the exit code of the step and therefore of this job.
#
# Why one plain step: Slurm 25.05 (LUMI since 2026-09-02) gives only 7 of 8 GCDs
# to --overlap steps that cover a subset of a multi-node job. A single non-overlap
# step spanning all nodes is not affected. ray_symmetric.sh asserts 8 GCDs anyway.
#
# Scaling: change --nodes and set POLICY_NODES / INFER_NODES / TP for the training
# script (see run_gsm8k_2node.sh). Nothing else here changes.

set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export SANDBOX="$ROOT/multitorch_sandbox"
export SCRIPTS="$ROOT/scripts"
export BINDS="/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038"
export GPUS_PER_NODE=8
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RAY_PORT=6379

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"

# The batch script runs on the first node of the allocation. The Ray head must be
# this node because we read its hsn0 address locally below.
[ "${SLURMD_NODENAME:-$(hostname)}" = "$HEAD_NODE" ] || {
  echo "FATAL: batch host ${SLURMD_NODENAME:-$(hostname)} != first node $HEAD_NODE" >&2
  exit 1
}

# Ray control traffic on the Slingshot fabric, not the management network.
# symmetric-run compares this address against every local interface to elect the
# head, and passes it to `ray start --head --node-ip-address=...`.
HEAD_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
echo "nodes: ${NODES[*]}   head: $HEAD_NODE ($RAY_ADDRESS)"

# --label      : prefix every output line with the task rank, so it is immediately
#                visible which task printed "On head node" (must be exactly one).
# --kill-on-bad-exit : any task failing (GPU assertion, ray start crash, timeout)
#                kills the whole step instead of leaving the others waiting.
srun --label --kill-on-bad-exit=1 \
     --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
     --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     bash "$SCRIPTS/ray_symmetric.sh"
