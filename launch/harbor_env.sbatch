#!/bin/bash
#SBATCH --ntasks-per-node=1
# Everything else (account, partition, nodes, cpus, mem, time, output) comes from the run's
# harbor_env.sbatch_args (config section harbor_env.slurm). Started by ./submit.sh for harbor
# configs, before the training job, which depends on it (--dependency=after) and cancels it on exit.
#
# Harbor environment plane for SkyRL on LUMI. One file, two roles, selected by the second argument.
# Everything here runs on the host (no container for this script itself; worker.py starts the
# singularity instances for the trials).
#
#   batch   (default) sbatch runs this on the first node. Starts the bridge server on that node,
#           publishes its URL to $BRIDGE_FILE (the training job reads it from there), then starts
#           one `worker` task per node with srun and blocks for the job lifetime. The URL file is
#           removed when the job ends for any reason.
#   worker  one task per node (node 0 included). Runs worker.py, which owns the singularity
#           instances on that node and serves trials to the bridge.
#
# Total workers (nodes x harbor_env.workers_per_node) must be >= generator.rate_limit.max_concurrency;
# render.py checks that.
#
# LUMI disables fakeroot (no subuid mapping, user namespaces blocked), hence BRIDGE_USE_FAKEROOT=0
# (harbor_env.use_fakeroot): "auto" would extract every SIF to a sandbox, fail, and retry without
# root on every start.

load_run() {
  RUN_DIR="${1:?usage: harbor_env.sbatch <run_dir> [batch|worker]}"
  # shellcheck disable=SC1091
  set -a; . "$RUN_DIR/job.env"; set +a
  SELF="$CODE_DIR/launch/harbor_env.sbatch"
}

# --- role: batch. Runs under sbatch on the first node of the allocation. ---
role_batch() {
  mkdir -p "$HARBOR_SIF_CACHE" "$SINGULARITY_CACHEDIR"

  # 1. bridge, host side of node 0 (this batch host). Advertise the Slingshot address so the
  #    training job (and the workers on other nodes) reach it over hsn, not the management network.
  local bridge_ip log="$RUN_DIR/harbor_bridge-$SLURM_JOB_ID.log"
  bridge_ip=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
  export BRIDGE_URL="http://$bridge_ip:$HARBOR_BRIDGE_PORT"
  ( module load cray-python && exec python3 "$HARBOR_APPTAINER_DIR/server.py" --port "$HARBOR_BRIDGE_PORT" ) > "$log" 2>&1 &
  sleep 3
  curl -sf "$BRIDGE_URL/status" >/dev/null || { echo "FATAL: bridge failed to start, see $log" >&2; exit 1; }

  # 2. publish the address; withdraw it whenever this job ends for any reason.
  echo "$BRIDGE_URL" > "$BRIDGE_FILE"
  trap 'rm -f "$BRIDGE_FILE"' EXIT
  trap 'exit 143' TERM INT HUP
  echo "bridge $BRIDGE_URL (job $SLURM_JOB_ID) published to $BRIDGE_FILE"
  echo "nodes: $SLURM_JOB_NUM_NODES x $HARBOR_WORKERS_PER_NODE workers = $(( SLURM_JOB_NUM_NODES * HARBOR_WORKERS_PER_NODE )) total"

  # 3. one worker process per node (node 0 included), blocks for the job lifetime.
  #    --kill-on-bad-exit: a failed worker ends the job, so a stale bridge.url is never left
  #    pointing at a half-dead environment plane.
  srun --label --kill-on-bad-exit=1 \
       --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
       --cpus-per-task="$SLURM_CPUS_PER_TASK" --export=ALL \
       bash "$SELF" "$RUN_DIR" worker
}

# --- role: worker. One srun task per node, on the host. ---
role_worker() {
  module load cray-python                      # host python3 is 3.6; worker needs >=3.7
  echo "[$SLURMD_NODENAME] worker x$HARBOR_WORKERS_PER_NODE -> $BRIDGE_URL"
  exec python3 "$HARBOR_APPTAINER_DIR/worker.py" --bridge-url "$BRIDGE_URL" \
       --sif-cache "$HARBOR_SIF_CACHE" --num-workers "$HARBOR_WORKERS_PER_NODE"
}

# --- dispatch ---
set -euo pipefail
load_run "${1:-}"
case "${2:-batch}" in
  batch)  role_batch ;;
  worker) role_worker ;;
  *)      echo "usage: harbor_env.sbatch <run_dir> [batch|worker]" >&2; exit 2 ;;
esac
