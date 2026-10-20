#!/bin/bash
#SBATCH --job-name=skyrl-tildeopen-gsm8k
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g       
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export BINDS="/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038"
export SANDBOX="$ROOT/multitorch_sandbox.sif"
export SCRIPTS="$ROOT/scripts"
export GPUS_PER_NODE=8
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RAY_PORT=6379

export MODEL_SRC="$ROOT/tildeopen"

# ---------------------------------------------------------------------------
# Run archive: snapshot the scripts now, copy the log when the job ends.
#
# --output=%x-%j.out is relative to the directory sbatch was run from, so the
# log lives at $SLURM_SUBMIT_DIR/<jobname>-<jobid>.out. Slurm keeps appending
# to it for the whole job, so a copy taken now would be nearly empty. Instead
# the copy is done in an EXIT trap, which runs after srun returns, whether the
# job finished, failed (set -e), or was cancelled / timed out (the TERM trap
# turns the signal into an exit so the EXIT trap still fires).
# ---------------------------------------------------------------------------
RUN_DIR="$ROOT/previous_runs/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
LOG_FILE="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
mkdir -p "$RUN_DIR"
cp -r "$SCRIPTS"/. "$RUN_DIR/"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/slurm_job.txt"
echo "run archive: $RUN_DIR"

LATEST_DIR="$ROOT/previous_runs/latest_run"
rm -rf "$LATEST_DIR"
mkdir -p "$LATEST_DIR"
for f in "$RUN_DIR"/*; do
  ln -s "$f" "$LATEST_DIR/$(basename "$f")"
done
ln -s "$LOG_FILE" "$LATEST_DIR/$(basename "$LOG_FILE")"
ln -s "$RUN_DIR"  "$LATEST_DIR/run_dir"

archive_log() {
  local rc=$?
  echo "job finished with exit code $rc, copying log to $RUN_DIR"
  cp -f "$LOG_FILE" "$RUN_DIR/" 2>/dev/null || true
  exit "$rc"
}
trap archive_log EXIT
trap 'exit 143' TERM INT HUP

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"

# The batch script runs on the first node of the allocation, which is also our Ray
# head node -- assert that so the local hsn0 lookup below can't silently be wrong.
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
# setting mem=0 is important, especially when running on dev-g (OOM otherwise)
srun --label --kill-on-bad-exit=1 --mem=0 \
     --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
     --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     bash "$SCRIPTS/ray_symmetric.sh"