#!/bin/bash
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --mem=0                       # important on dev-g, OOM otherwise
#SBATCH --exclusive
# Everything else (account, partition, nodes, time, GPUs, output) comes from the run's sbatch_args,
# written by launch/render.py. Do not sbatch this file directly: use ./submit.sh <config>.
#
# SkyRL + FSDP on LUMI, for every task. One file, five roles, selected by the second argument:
#
#   batch   (default) sbatch runs this on the first node of the allocation. For Harbor, waits for
#           the environment plane's bridge. Computes RAY_ADDRESS, starts one `node` task per node.
#   node    one task per node, on the host (outside the container). Asserts the GPU count,
#           stages the checkpoint to /tmp, enters the container and runs `inner`.
#           On the last JUDGE_NODES nodes (configs with a judge: section): stages the judge model
#           and runs `judge` instead.
#   judge   inside the container, on a judge node, outside Ray. Starts GPUS_PER_NODE / JUDGE_TP vLLM
#           servers with the judge model, publishes their URLs to $RUN_DIR/judge_urls/<node>.txt once
#           they are healthy (the driver waits for them: skyrl_extensions/judge/env.py), serves until
#           the driver has ended. A server that dies ends the job.
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
# Hang watchdog (batch node, background). A stuck collective never returns and never errors: the job would sit
# silent until the time limit with every node billed (jobs 23677895 and 23681198 on nid[005018-005021], hung in
# the first forward pass; the worker timeout is 1 h because rank 0 needs minutes to load the model).
# Once training has started, the Slurm log gets a line at least every step; the longest quiet phases are a
# long step, a checkpoint and the HF export (each well under 10 min for TildeOpen-30B). No new line for
# HANG_TIMEOUT_MIN -> record the nodes in $RUN_DIR/HUNG, print how to resubmit, cancel the job.
# watchdog <log> <limit seconds> <poll seconds>
watchdog() {
  local log=$1 limit=$2 poll=$3 age
  until grep -q "Started: 'step'" "$log" 2>/dev/null; do sleep "$poll"; done
  while sleep "$poll"; do
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    (( age > limit )) || continue
    {
      echo "WATCHDOG: no output for $(( age / 60 )) min after training started: the job hangs. Cancelling it."
      echo "  nodes: ${SLURM_JOB_NODELIST:-?}"
      echo "  stacks of a hung job (before cancelling by hand): fsdp-skyrl-harbor/inspect_hung_job.sh <jobid>"
      echo "  resubmit without these nodes: add  'slurm.extra=[\"--exclude=${SLURM_JOB_NODELIST:-}\"]'"
      echo "  and, if a checkpoint exists, skyrl.trainer.resume_mode=latest"
    } | tee "$(dirname "$log")/HUNG"
    scancel "${SLURM_JOB_ID:?}"
    return
  done
}

role_batch() {
  scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/slurm_job.txt"
  watchdog "$RUN_DIR/slurm-$SLURM_JOB_ID.out" $(( ${HANG_TIMEOUT_MIN:-15} * 60 )) 60 &
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
  # The last JUDGE_NODES nodes serve the judge model and do not join Ray (never the head: it is first).
  JUDGE_NODELIST=$(IFS=,; echo "${NODES[*]:RAY_NODES}")
  export JUDGE_NODELIST
  rm -rf "$RUN_DIR/judge_urls" "$RUN_DIR/train.done"
  if (( JUDGE_NODES )); then
    mkdir -p "$RUN_DIR/judge_urls" "$RUN_DIR/judge"
    echo "judge nodes: $JUDGE_NODELIST ($JUDGE_SERVED_NAME, logs in $RUN_DIR/judge)"
  fi

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
  # on Lustre. MODEL (the staged path) is what the SkyRL args point at. A judge node needs only
  # the judge model.
  local role=inner src=$MODEL_SRC dst=$MODEL
  if [[ ",${JUDGE_NODELIST:-}," == *",$SLURMD_NODENAME,"* ]]; then
    role=judge src=$JUDGE_MODEL_SRC dst=$JUDGE_MODEL
  fi
  mkdir -p "$dst"
  echo "[$SLURMD_NODENAME] staging $src -> $dst, timing on the next line:"
  TIMEFORMAT="[$SLURMD_NODENAME] staging took %R s"
  time (ls "$src" | xargs -P 16 -I{} cp -rL "$src/{}" "$dst/")

  # Host RAM of this node every 5 s -> $RUN_DIR/mem/<node>.log (GB). The node is exclusive, so /proc/meminfo is
  # the job: used = total - available, anon = process memory, shmem = /tmp + /dev/shm (staged model, ray object
  # store), file = page cache incl. dirty Lustre pages. Answers "what filled the node" after a host OOM.
  mkdir -p "$RUN_DIR/mem"
  ( while sleep 5; do
      awk -v t="$(date +%T)" '{v[$1]=$2/1048576} END{printf "%s used %.0f anon %.0f shmem %.0f file %.0f dirty %.1f\n", t,
        v["MemTotal:"]-v["MemAvailable:"], v["AnonPages:"], v["Shmem:"], v["Cached:"]-v["Shmem:"], v["Dirty:"]+v["Writeback:"]}' /proc/meminfo
    done >> "$RUN_DIR/mem/$SLURMD_NODENAME.log" 2>/dev/null ) &

  exec singularity exec -B "$BINDS" "$SANDBOX" bash "$SELF" "$RUN_DIR" "$role"
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
    --min-nodes "$RAY_NODES" \
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
  (( JUDGE_NODES )) || exec python -m "$ENTRYPOINT" "${args[@]}" "$@"
  # With judge nodes: tell them the driver has ended, whatever the result, so their srun tasks return
  # and the job ends instead of serving the judge until the time limit.
  local rc=0
  python -m "$ENTRYPOINT" "${args[@]}" "$@" || rc=$?
  touch "$RUN_DIR/train.done"
  exit "$rc"
}

# --- role: judge. Inside the container, one per judge node, outside Ray. ---
# Same pattern as rl-rollgen's lumi/_vllm_node.sh: independent `vllm serve` processes, GPUS_PER_NODE / JUDGE_TP
# per node, each on its own GPUs and port. No router: the env spreads its requests over the published URLs.
# The servers have no API key: anyone on the LUMI network can reach them while the job runs.
role_judge() {
  apply_env
  local node=${SLURMD_NODENAME:-$(hostname)} n=$(( GPUS_PER_NODE / JUDGE_TP )) i addr tmp
  # vLLM's torch.compile / Triton caches default to ~/.cache: the LUMI home quota is 100K files.
  tmp="/tmp/$USER-judge-cache-${SLURM_JOB_ID:-x}"
  export XDG_CACHE_HOME="$tmp" VLLM_CACHE_ROOT="$tmp/vllm" TORCHINDUCTOR_CACHE_DIR="$tmp/inductor" TRITON_CACHE_DIR="$tmp/triton"
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  # Address the other nodes reach this one at: hsn0 (the container has no `ip`; SIOCGIFADDR = 0x8915).
  addr=$(python -c 'import fcntl, socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", b"hsn0"))[20:24]))') \
    || die "[$node] no hsn0 address"

  local gpus
  pids=()   # global: the EXIT trap runs after this function has returned
  mapfile -t gpus < <(tr ',' '\n' <<< "$HIP_VISIBLE_DEVICES" | grep .)
  trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
  trap 'exit 143' TERM INT HUP
  for (( i = 0; i < n; i++ )); do
    echo "[$node] judge server $i: GPUs $(IFS=,; echo "${gpus[*]:i*JUDGE_TP:JUDGE_TP}"), port $(( JUDGE_PORT + i ))"
    # shellcheck disable=SC2086
    HIP_VISIBLE_DEVICES=$(IFS=,; echo "${gpus[*]:i*JUDGE_TP:JUDGE_TP}") vllm serve "$JUDGE_MODEL" \
      --served-model-name "$JUDGE_SERVED_NAME" --tensor-parallel-size "$JUDGE_TP" \
      --host 0.0.0.0 --port $(( JUDGE_PORT + i )) $JUDGE_VLLM_ARGS > "$RUN_DIR/judge/$node-vllm-$i.log" 2>&1 &
    pids+=($!)
  done

  healthy() { python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "$1" 2>/dev/null; }
  SECONDS=0
  for (( i = 0; i < n; i++ )); do
    until healthy "http://127.0.0.1:$(( JUDGE_PORT + i ))/health"; do
      kill -0 "${pids[i]}" 2>/dev/null || die "[$node] judge server $i died during startup, see $RUN_DIR/judge/$node-vllm-$i.log"
      (( SECONDS < JUDGE_STARTUP_TIMEOUT_S )) || die "[$node] judge server $i not healthy after ${JUDGE_STARTUP_TIMEOUT_S}s"
      sleep 10
    done
  done
  for (( i = 0; i < n; i++ )); do echo "http://$addr:$(( JUDGE_PORT + i ))/v1"; done > "$RUN_DIR/judge_urls/$node.tmp"
  mv "$RUN_DIR/judge_urls/$node.tmp" "$RUN_DIR/judge_urls/$node.txt"   # atomic: the driver never reads a partial file
  echo "[$node] $n judge server(s) healthy after ${SECONDS}s at $addr"

  # Serve until the driver has ended (role train touches train.done). A dead server ends the job
  # (srun --kill-on-bad-exit): the env would otherwise retry against it and then fail the run anyway.
  until [ -e "$RUN_DIR/train.done" ]; do
    for (( i = 0; i < n; i++ )); do
      kill -0 "${pids[i]}" 2>/dev/null || die "[$node] judge server $i exited, see $RUN_DIR/judge/$node-vllm-$i.log"
    done
    sleep 10
  done
  echo "[$node] driver ended: stopping the judge servers"
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
  judge) role_judge ;;
  train) role_train "$@" ;;
  *)     die "usage: job.sbatch <run_dir> [batch|node|inner|judge|train]   (or: source job.sbatch <run_dir>)" ;;
esac
