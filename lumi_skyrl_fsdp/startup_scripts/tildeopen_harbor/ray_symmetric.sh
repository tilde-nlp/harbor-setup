#!/bin/bash
# Runs on the HOST side, once per node, as one task of the srun step in
# submit_2node.sbatch. Enters the singularity sandbox and runs `ray symmetric-run`,
# which decides by itself whether this node is the head or a worker.
#
# Expects from the sbatch environment:
#   ROOT SANDBOX SCRIPTS BINDS GPUS_PER_NODE RAY_ADDRESS SLURM_JOB_NUM_NODES SLURM_CPUS_PER_TASK
set -euo pipefail

# GPU assertion must happen here, on the host side, before skyrl_env.sh unsets
# ROCR_VISIBLE_DEVICES inside the container.
NGPU=$(tr ',' '\n' <<< "${ROCR_VISIBLE_DEVICES:-}" | grep -c .)
[ "$NGPU" -eq "$GPUS_PER_NODE" ] || {
  echo "[$SLURMD_NODENAME] FATAL: step got $NGPU/$GPUS_PER_NODE GPUs (ROCR=${ROCR_VISIBLE_DEVICES:-unset})" >&2
  exit 1
}

# Diagnostic: the `src` address in this line is what a worker raylet will bind to
# (ray start derives it from the route to the head). Expect an hsn* interface.
echo "[$SLURMD_NODENAME] route to head: $(ip route get "${RAY_ADDRESS%%:*}" | head -1)"

# Stage the checkpoint into node-local RAM (/tmp): sequential reads instead of mmap page faults on Lustre.
if [ -d "$MODEL_SRC" ]; then
  SRC="${MODEL_SRC%/}"
else
  SRC=$(ls -d "$ROOT/hf/hub/models--${MODEL_SRC//\//--}"/snapshots/*/ | head -1)
  SRC="${SRC%/}"
fi
export MODEL="/tmp/$USER-model/$(basename "$MODEL_SRC")"
mkdir -p "$MODEL"
echo "[$SLURMD_NODENAME] staging $SRC -> $MODEL, timing on the next line:"
TIMEFORMAT="[$SLURMD_NODENAME] staging took %R s"
time (ls "$SRC" | xargs -P 16 -I{} cp -rL "$SRC/{}" "$MODEL/")

# Not allowed with symmetric-run: --head, --node-ip-address, --port, --block.
# It adds those itself. Everything else is forwarded to `ray start` on every node.
exec singularity exec -B "$BINDS" "$SANDBOX" bash -c "
  source $SCRIPTS/skyrl_env.sh
  ray stop --force >/dev/null 2>&1 || true
  exec ray symmetric-run \
    --address $RAY_ADDRESS \
    --min-nodes $SLURM_JOB_NUM_NODES \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --num-gpus=$GPUS_PER_NODE \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    -- bash $SCRIPTS/run_training.sh"