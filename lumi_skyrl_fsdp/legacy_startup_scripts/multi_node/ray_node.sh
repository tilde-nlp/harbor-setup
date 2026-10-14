#!/bin/bash
# Runs on the HOST side of one srun step. Starts exactly one Ray node (head or worker)
# inside the singularity sandbox and blocks for the lifetime of the job.
#   bash ray_node.sh head|worker
# Expects ROOT, SANDBOX, SCRIPTS, RAY_PORT, RAY_ADDRESS, GPUS_PER_NODE from the sbatch env.
set -euo pipefail

ROLE="${1:?usage: ray_node.sh head|worker}"

# This node's Slingshot address (same trick as for the head in the sbatch script).
MY_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)

NGPU=$(tr ',' '\n' <<< "${ROCR_VISIBLE_DEVICES:-}" | grep -c .)
[ "$NGPU" -eq "$GPUS_PER_NODE" ] || {
  echo "[$SLURMD_NODENAME] FATAL: step got $NGPU/$GPUS_PER_NODE GPUs (ROCR=$ROCR_VISIBLE_DEVICES)" >&2
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
    RAY_CMD="ray start --address=$RAY_ADDRESS $COMMON"
    ;;
  *)
    echo "unknown role: $ROLE" >&2; exit 1
    ;;
esac

echo "[$SLURMD_NODENAME] $ROLE @ $MY_IP: $RAY_CMD"
exec singularity exec -B "${BINDS:-/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038}" "$SANDBOX" \
  bash -c "source $SCRIPTS/skyrl_env.sh && ray stop --force >/dev/null 2>&1; $RAY_CMD"
