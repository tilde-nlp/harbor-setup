#!/bin/bash
# Runs on the HOST side as one task of the multi-node Ray step (one task per node). Starts
# exactly one Ray node inside the singularity sandbox and blocks for the lifetime of the job.
# Task 0 (SLURM_PROCID=0, the first node of the allocation = the batch host) is the head;
# every other task is a worker. An explicit role argument still works: bash ray_node.sh worker
# Expects ROOT, SANDBOX, SCRIPTS, RAY_PORT, RAY_ADDRESS, GPUS_PER_NODE from the sbatch env.
set -euo pipefail

ROLE="${1:-$([ "${SLURM_PROCID:-0}" = 0 ] && echo head || echo worker)}"

# This node's Slingshot address (same trick as for the head in the sbatch script).
MY_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)

NGPU=$(tr ',' '\n' <<< "${ROCR_VISIBLE_DEVICES:-}" | grep -c .)
[ "$NGPU" -eq "$GPUS_PER_NODE" ] || {
  echo "[$SLURMD_NODENAME] FATAL: step got $NGPU/$GPUS_PER_NODE GPUs (ROCR=$ROCR_VISIBLE_DEVICES)" >&2
  exit 1
}

# Slingshot credentials. Every Ray worker on this node (policy ranks, vLLM TP workers) inherits
# them from this raylet; without SLINGSHOT_VNIS the cxi provider cannot open a domain and RCCL
# ends up on TCP sockets (or, with NCCL_NET pinned, fails). Fail here, in the first minute.
echo "[$SLURMD_NODENAME] SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-<unset>} SLINGSHOT_DEVICES=${SLINGSHOT_DEVICES:-<unset>} SLINGSHOT_SVC_IDS=${SLINGSHOT_SVC_IDS:-<unset>}"
[ -n "${SLINGSHOT_VNIS:-}" ] || {
  echo "[$SLURMD_NODENAME] FATAL: no Slingshot VNI in this step - is this a single-node step? (see submit_2node.sbatch)" >&2
  exit 1
}

# Ray workers on this node inherit the environment of THIS raylet, not the driver's,
# which is why skyrl_env.sh is sourced here and not only in the driver step.
COMMON="--node-ip-address=$MY_IP --num-gpus=$GPUS_PER_NODE --num-cpus=${SLURM_CPUS_PER_TASK:-56} --block"

case "$ROLE" in
  head)
    RAY_CMD="ray start --head --port=$RAY_PORT --dashboard-host=0.0.0.0 --dashboard-port=8265 $COMMON"
    ;;
  worker)
    # Same step as the head, so wait until its GCS port answers before joining (up to ~3 min).
    HEAD_HOST=${RAY_ADDRESS%:*}
    for _ in $(seq 90); do (exec 3<>"/dev/tcp/$HEAD_HOST/$RAY_PORT") 2>/dev/null && break; sleep 2; done
    (exec 3<>"/dev/tcp/$HEAD_HOST/$RAY_PORT") 2>/dev/null || { echo "[$SLURMD_NODENAME] FATAL: Ray head $RAY_ADDRESS never came up" >&2; exit 1; }
    RAY_CMD="ray start --address=$RAY_ADDRESS $COMMON"
    ;;
  *)
    echo "unknown role: $ROLE" >&2; exit 1
    ;;
esac

echo "[$SLURMD_NODENAME] $ROLE @ $MY_IP: $RAY_CMD"
exec singularity exec -B "${BINDS:-/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038}" "$SANDBOX" \
  bash -c "source $SCRIPTS/skyrl_env.sh && ray stop --force >/dev/null 2>&1; $RAY_CMD"