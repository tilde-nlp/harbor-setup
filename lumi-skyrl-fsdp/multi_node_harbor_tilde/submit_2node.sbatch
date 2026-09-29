#!/bin/bash
#SBATCH --job-name=skyrl-gsm8k-14b
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g        # dev-g works too if its node/time limits allow 2 nodes
#SBATCH --nodes=4
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
# NOTE on Slingshot VNIs (jobs 22207xxx/22210806, 2026-09-21): Slurm's hpe_slingshot plugin
# hands a step its network VNI and the SLINGSHOT_VNIS/SLINGSHOT_SVC_IDS environment only for
# MULTI-NODE steps. The old layout ran one single-node srun step per Ray node, so no Ray
# process ever had Slingshot credentials: libfabric's cxi provider failed fi_domain() with
# ENOSYS ("Couldn't create domain. RC: -38") and RCCL silently fell back to TCP sockets.
# --network=job_vni did not change that here (site policy). So all Ray nodes now run as ONE
# plain multi-node step: task 0 is the head, the rest are workers (ray_node.sh picks the role
# from SLURM_PROCID). One step = one VNI shared by every rank, which is also what lets the
# trainer's rank 0 and the vLLM workers on other nodes form an RCCL group.
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

BRIDGE_FILE=$ROOT/bridge.url
for _ in $(seq 60); do [ -s "$BRIDGE_FILE" ] && break; sleep 5; done   # up to 5 min
[ -s "$BRIDGE_FILE" ] || { echo "FATAL: no $BRIDGE_FILE - submit_harbor_env.sbatch not running?" >&2; exit 1; }
export APPTAINER_BRIDGE_URL=$(cat "$BRIDGE_FILE")
curl -sf --max-time 5 "$APPTAINER_BRIDGE_URL/status" \
  || { echo "FATAL: bridge $APPTAINER_BRIDGE_URL unreachable from $SLURMD_NODENAME" >&2; exit 1; }
echo "harbor bridge: $APPTAINER_BRIDGE_URL"



# 1+2. One plain multi-node step for all Ray nodes (see the VNI note above). Each task owns
#      its whole node: 8 GCDs, 56 cores. ray_node.sh asserts the GPU count and the VNI, starts
#      the head on task 0 and a worker (which waits for the head) everywhere else, and blocks.
srun --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
     --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     bash "$SCRIPTS/ray_node.sh" &
RAY_STEP_PID=$!
sleep 30
kill -0 "$RAY_STEP_PID" 2>/dev/null || { echo "FATAL: the Ray step exited during startup (see the [nidXXXXXX] lines above)" >&2; exit 1; }

# 3. Driver: runs here, inside the batch step (which holds all 8 GCDs and all
#    cores on this node -- no srun, no --overlap). The driver only talks to the
#    Ray head over TCP; GPU work happens in actors under the raylets, and those
#    inherit the raylet's environment, not the driver's.
#    run_harbor_2node.sh first waits until all GPUs have joined the cluster.
export BINDS="/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038"
singularity exec -B "$BINDS" "$SANDBOX" \
  bash -c "source $SCRIPTS/skyrl_env.sh && bash $SCRIPTS/run_harbor_2node.sh"

# When the driver returns, this script exits and Slurm tears down the Ray step on all
# nodes, so nothing survives into the next job.
