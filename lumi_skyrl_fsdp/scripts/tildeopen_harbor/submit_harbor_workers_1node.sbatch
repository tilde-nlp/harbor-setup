#!/bin/bash
#SBATCH --job-name=harbor-env
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g             # smoke: 1 shared node. Scale: --partition=standard, whole nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/logs/harbor-env-%j.out
#
# Harbor environment plane for SkyRL on LUMI. One file, two roles, selected by the first argument.
# Everything here runs on the host (no container for this script itself; worker.py starts the
# singularity instances for the trials).
#
#   (none)  sbatch runs this on the first node. Starts the bridge server on that node, publishes
#           its URL to $BRIDGE_FILE (the SkyRL job reads it from there), then starts one `worker`
#           task per node with srun and blocks for the job lifetime. The URL file is removed when
#           the job ends for any reason.
#   worker  one task per node (node 0 included). Runs worker.py, which owns the singularity
#           instances on that node and serves trials to the bridge.
#
# Submit:   sbatch harbor_env.sh
# Scale:    WORKERS_PER_NODE=16 sbatch --nodes=4 --partition=standard harbor_env.sh
# Every variable in section 1 uses ": ${VAR:=default}", so it can be overridden by exporting it
# before sbatch. Total workers (nodes x WORKERS_PER_NODE) must be >= SkyRL's
# generator.rate_limit.max_concurrency (N_CONCURRENT in skyrl.sh).

# =====================================================================
# 1. CONFIG
# =====================================================================
load_config() {
  set -a
  : "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
  : "${WORKERS_PER_NODE:=8}"
  : "${BRIDGE_PORT:=9910}"
  : "${BRIDGE_FILE:=$ROOT/bridge.url}"           # read by skyrl.sh at startup
  : "${HARBOR_APPT:=$ROOT/marin-harbor/src/harbor/environments/apptainer}"   # server.py, worker.py
  : "${SIF_CACHE:=$ROOT/sif_cache}"
  : "${SINGULARITY_CACHEDIR:=$ROOT/singularity_cache}"
  : "${LOG_DIR:=$ROOT/logs}"
  # LUMI disables fakeroot (no subuid mapping, user namespaces blocked). "auto" would extract
  # every SIF to a sandbox, fail, and retry without root on every start.
  : "${BRIDGE_USE_FAKEROOT:=0}"
  set +a
}

# =====================================================================
# 2. ROLES
# =====================================================================

# --- role: batch. Runs under sbatch on the first node of the allocation. ---
role_batch() {
  mkdir -p "$SIF_CACHE" "$SINGULARITY_CACHEDIR" "$LOG_DIR"

  # Slurm executes a private copy of the batch script from /var/spool/slurmd on this node only.
  # The other nodes need the file on /scratch, so the `worker` role runs from this copy.
  export SELF="$LOG_DIR/harbor-env-${SLURM_JOB_ID}.sh"
  cp "${BASH_SOURCE[0]}" "$SELF"

  # 1. bridge, host side of node 0 (this batch host). Advertise the Slingshot address so the
  #    SkyRL job (and the workers on other nodes) reach it over hsn, not the management network.
  local bridge_ip
  bridge_ip=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
  export BRIDGE_URL="http://$bridge_ip:$BRIDGE_PORT"
  ( module load cray-python && exec python3 "$HARBOR_APPT/server.py" --port "$BRIDGE_PORT" ) \
      > "$LOG_DIR/bridge-$SLURM_JOB_ID.log" 2>&1 &
  sleep 3
  curl -sf "$BRIDGE_URL/status" >/dev/null || { echo "FATAL: bridge failed to start, see $LOG_DIR/bridge-$SLURM_JOB_ID.log" >&2; exit 1; }

  # 2. publish the address; withdraw it whenever this job ends for any reason.
  echo "$BRIDGE_URL" > "$BRIDGE_FILE"
  trap 'rm -f "$BRIDGE_FILE"' EXIT
  trap 'exit 143' TERM INT HUP
  echo "bridge $BRIDGE_URL (job $SLURM_JOB_ID) published to $BRIDGE_FILE"
  echo "nodes: $SLURM_JOB_NUM_NODES x $WORKERS_PER_NODE workers = $(( SLURM_JOB_NUM_NODES * WORKERS_PER_NODE )) total"

  # 3. one worker process per node (node 0 included), blocks for the job lifetime.
  #    --kill-on-bad-exit: a failed worker ends the job, so a stale bridge.url is never left
  #    pointing at a half-dead environment plane.
  srun --label --kill-on-bad-exit=1 \
       --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
       --cpus-per-task="$SLURM_CPUS_PER_TASK" --export=ALL \
       bash "$SELF" worker
}

# --- role: worker. One srun task per node, on the host. ---
role_worker() {
  module load cray-python                      # host python3 is 3.6; worker needs >=3.7
  echo "[$SLURMD_NODENAME] worker x$WORKERS_PER_NODE -> $BRIDGE_URL"
  exec python3 "$HARBOR_APPT/worker.py" --bridge-url "$BRIDGE_URL" \
       --sif-cache "$SIF_CACHE" --num-workers "$WORKERS_PER_NODE"
}

# =====================================================================
# 3. DISPATCH
# =====================================================================
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  load_config
  return 0
fi

set -euo pipefail
load_config

case "${1:-batch}" in
  batch)  role_batch ;;
  worker) role_worker ;;
  *)      echo "usage: sbatch $0   (or: bash $0 worker)" >&2; exit 2 ;;
esac
