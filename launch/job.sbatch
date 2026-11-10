#!/bin/bash
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --mem=0                       # important on dev-g, OOM otherwise
#SBATCH --exclusive
# Everything else (account, partition, nodes, time, GPUs, output) comes from the run's sbatch_args,
# written by launch/render.py. Do not sbatch this file directly: use ./submit.sh <config>.
#
# SkyRL + FSDP on LUMI, for every task. One file, four roles, selected by the second argument:
#
#   batch   (default) sbatch runs this on the first node of the allocation. For Harbor, waits for
#           the environment plane's bridge. Computes RAY_ADDRESS, starts one `node` task per node.
#   node    one task per node, on the host (outside the container). Asserts the GPU count,
#           stages the checkpoint to /tmp, enters the container and runs `inner`.
#   inner   inside the container. Applies launch/lumi_env.sh and the config's env:, runs
#           `ray symmetric-run`, which elects head/workers and runs `train` on the head once every
#           node joined.
#   train   the driver, on the head node, inside the container, under ray:
#           python -m $ENTRYPOINT $(cat $RUN_DIR/skyrl_args)
#
# Every role reads $RUN_DIR/job.env (paths, layout, entrypoint), so the run is frozen at submit
# time: the job runs the code snapshot in $RUN_DIR/code, editing the repo afterwards changes nothing.
#
# Debugging inside the container, with the exact environment of a run:
#     singularity exec -B "$BINDS" "$SANDBOX" bash
#     source <run_dir>/code/launch/job.sbatch <run_dir>      # loads the environment, runs nothing

load_run() {
  RUN_DIR="${1:?usage: job.sbatch <run_dir> [batch|node|inner|train]}"
  # shellcheck disable=SC1091
  set -a; . "$RUN_DIR/job.env"; set +a
  SELF="$CODE_DIR/launch/job.sbatch"
}

apply_env() {
  # shellcheck disable=SC1091
  . "$CODE_DIR/launch/lumi_env.sh"
  # shellcheck disable=SC1091
  . "$RUN_DIR/env.sh"
  if [ -s "$RUN_DIR/env.sh" ] && grep -q '^export' "$RUN_DIR/env.sh"; then
    echo "[$(hostname)] config env: $(grep '^export' "$RUN_DIR/env.sh" | cut -d' ' -f2- | tr '\n' ' ')"
  fi
}

die() { echo "FATAL: $*" >&2; exit 1; }

# --- role: batch. Runs under sbatch on the first node of the allocation. ---
role_batch() {
  scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/slurm_job.txt"
  echo "run dir: $RUN_DIR"
  echo "$LAYOUT_SUMMARY"

  # TERM (scancel, time limit) becomes a normal exit, so the EXIT trap below always runs.
  trap 'exit 143' TERM INT HUP
  if (( HARBOR )); then
    # The Harbor environment plane (launch/harbor_env.sbatch) is its own job. submit.sh starts it
    # first and records its id; it ends with this job, whatever the reason.
    if [ -s "$RUN_DIR/harbor_env.jobid" ]; then
      trap 'echo "stopping harbor env job $(cat "$RUN_DIR/harbor_env.jobid")"; scancel "$(cat "$RUN_DIR/harbor_env.jobid")"' EXIT
    fi
    # The env plane publishes its bridge address to $BRIDGE_FILE. Wait for it (up to 5 min), check
    # it answers, export the URL so the node tasks, the container and the driver inherit it.
    for _ in $(seq 60); do [ -s "$BRIDGE_FILE" ] && break; sleep 5; done
    [ -s "$BRIDGE_FILE" ] || die "no $BRIDGE_FILE after 5 min - harbor env job not running? see $RUN_DIR/harbor_env-*.out"
    APPTAINER_BRIDGE_URL=$(cat "$BRIDGE_FILE")
    export APPTAINER_BRIDGE_URL
    curl -sf --max-time 5 "$APPTAINER_BRIDGE_URL/status" >/dev/null \
      || die "bridge $APPTAINER_BRIDGE_URL unreachable from $SLURMD_NODENAME"
    echo "harbor bridge: $APPTAINER_BRIDGE_URL"
  fi

  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE="${NODES[0]}"
  # The batch script runs on the first node of the allocation, which is also our Ray head node.
  # Assert that so the local hsn0 lookup below can't silently be wrong.
  [ "${SLURMD_NODENAME:-$(hostname)}" = "$HEAD_NODE" ] \
    || die "batch host ${SLURMD_NODENAME:-$(hostname)} != first node $HEAD_NODE"
  (( SLURM_JOB_NUM_NODES == NUM_NODES )) \
    || die "allocation has $SLURM_JOB_NUM_NODES nodes, the config's layout needs $NUM_NODES"

  # Ray control traffic on the Slingshot fabric, not the management network. symmetric-run
  # compares this address against every local interface to elect the head, and passes it to
  # `ray start --head --node-ip-address=...`.
  HEAD_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
  export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
  echo "nodes: ${NODES[*]}   head: $HEAD_NODE ($RAY_ADDRESS)"

  # --label            : prefix every output line with the task rank, so it is immediately
  #                      visible which task printed "On head node" (must be exactly one).
  # --kill-on-bad-exit : any task failing (GPU assertion, ray start crash, timeout) kills the
  #                      whole step instead of leaving the others waiting.
  # --mem=0 matters on dev-g (OOM otherwise).
  srun --label --kill-on-bad-exit=1 --mem=0 \
       --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
       --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
       bash "$SELF" "$RUN_DIR" node
}

# --- role: node. One srun task per node, on the host, outside the container. ---
role_node() {
  # GPU assertion must happen here, on the host side, before lumi_env.sh unsets
  # ROCR_VISIBLE_DEVICES inside the container. (srun --overlap loses a GCD: notes/BUGS.md)
  local ngpu
  ngpu=$(tr ',' '\n' <<< "${ROCR_VISIBLE_DEVICES:-}" | grep -c .)
  (( ngpu == GPUS_PER_NODE )) \
    || die "[$SLURMD_NODENAME] step got $ngpu/$GPUS_PER_NODE GPUs (ROCR=${ROCR_VISIBLE_DEVICES:-unset})"

  # Diagnostic: the `src` address in this line is what a worker raylet will bind to
  # (ray start derives it from the route to the head). Expect an hsn* interface.
  echo "[$SLURMD_NODENAME] route to head: $(ip route get "${RAY_ADDRESS%%:*}" | head -1)"

  # Stage the checkpoint into node-local RAM (/tmp): sequential reads instead of mmap page faults
  # on Lustre. MODEL (the staged path) is what the SkyRL args point at.
  mkdir -p "$MODEL"
  echo "[$SLURMD_NODENAME] staging $MODEL_SRC -> $MODEL, timing on the next line:"
  TIMEFORMAT="[$SLURMD_NODENAME] staging took %R s"
  time (ls "$MODEL_SRC" | xargs -P 16 -I{} cp -rL "$MODEL_SRC/{}" "$MODEL/")

  exec singularity exec -B "$BINDS" "$SANDBOX" bash "$SELF" "$RUN_DIR" inner
}

# --- role: inner. Inside the container, one per node. ---
role_inner() {
  apply_env
  ray stop --force >/dev/null 2>&1 || true
  # Not allowed with symmetric-run: --head, --node-ip-address, --port, --block. It adds those
  # itself. Everything else is forwarded to `ray start` on every node. The command after `--`
  # runs on the head node only, once --min-nodes have joined.
  exec ray symmetric-run \
    --address "$RAY_ADDRESS" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus="$SLURM_CPUS_PER_TASK" \
    --num-gpus="$GPUS_PER_NODE" \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    -- bash "$SELF" "$RUN_DIR" train
}

# --- role: train. The driver, on the head node, inside the container. ---
role_train() {
  local args
  mapfile -t args < "$RUN_DIR/skyrl_args"
  echo "$LAYOUT_SUMMARY"
  echo "python -m $ENTRYPOINT  (${#args[@]} args from $RUN_DIR/skyrl_args)${ENTRYPOINT_CWD:+, cwd $ENTRYPOINT_CWD}"
  if [ -n "$ENTRYPOINT_CWD" ]; then cd "$ENTRYPOINT_CWD"; fi   # modules under examples.* live in /opt/SkyRL
  exec python -m "$ENTRYPOINT" "${args[@]}" "$@"
}

# --- dispatch ---
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # Sourced (debugging inside the container): load the run and the environment, run nothing,
  # leave the caller's shell options alone.
  load_run "$1"
  apply_env
  return 0
fi

set -euo pipefail
load_run "${1:-}"
role="${2:-batch}"
shift 2 2>/dev/null || shift $#
case "$role" in
  batch) role_batch ;;
  node)  role_node ;;
  inner) role_inner ;;
  train) role_train "$@" ;;
  *)     die "usage: job.sbatch <run_dir> [batch|node|inner|train]   (or: source job.sbatch <run_dir>)" ;;
esac
