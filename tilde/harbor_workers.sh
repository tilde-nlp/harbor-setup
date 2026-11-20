#!/bin/bash
# Docker sandboxes on Tilde for a SkyRL Harbor training job on LUMI (configs with harbor_env.sandboxes: tilde).
# Run it after ./submit.sh on LUMI, in tmux: it waits for the job's bridge, opens an ssh tunnel to it through a LUMI
# login node, and runs tilde/docker_worker.py (the trials' containers, in local Docker) until the LUMI job ends.
# A dropped tunnel is reopened; when it stops, the run's containers are removed.
#
#   tilde/harbor_workers.sh --lumi-run /scratch/.../previous_runs/<run>    # the run dir that submit.sh printed
#   tilde/harbor_workers.sh --lumi-run 12345678                           # or the training job's id (queued/running)
#
#   --lumi-run RUN         the run on LUMI: its run dir, or the training job's id (required)
#   --lumi-host HOST       LUMI login node for lookups and the tunnel (default: lumi, the ~/.ssh/config name)
#   --max-envs N           sandboxes at once (default 8). The job waits until the workers offer at least its
#                          generator.rate_limit.max_concurrency. Size by CPU and RAM: each sandbox gets the task's
#                          cpus and memory_mb (override_cpus / override_memory_mb in the config)
#   --num-workers N        threads running bridge jobs (default 32)
#   --wait MIN             max minutes to wait for the job to start and publish its bridge (default 180)
#   --no-resource-limits   containers without the task's --cpus / --memory
#   -h, --help             print this header
#
# Needs: docker without sudo, python3 >= 3.8, curl, and ssh to LUMI without a prompt: the key in an ssh-agent of
# this tmux window (eval "$(ssh-agent)"; ssh-add ~/.ssh/<lumi key>), as for rl-rollgen's tilde/harbor_run.sh.
# Before the first run: python3 tilde/docker_worker.py --selftest   (checks docker here, ~1 min)
#
# Logs: logs/tilde/<run>/worker.log (every job that failed, status every minute) and tunnel.log.
# Stop with Ctrl-C: the worker removes its containers. The LUMI job is not stopped: its trials fail (masked)
# until a new harbor_workers.sh connects; after 30 min without workers they raise (BRIDGE_WORKERS_DEAD_TIMEOUT).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
die() { echo "FATAL: $*" >&2; exit 1; }
say() { echo "$(date +%H:%M:%S) $*"; }

LUMI_RUN="" LUMI_HOST=lumi MAX_ENVS=8 NUM_WORKERS=32 WAIT_MIN=180 WORKER_ARGS=()
while (( $# )); do
  case "$1" in
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    --lumi-run)    LUMI_RUN=$2; shift 2 ;;
    --lumi-host)   LUMI_HOST=$2; shift 2 ;;
    --max-envs)    MAX_ENVS=$2; shift 2 ;;
    --num-workers) NUM_WORKERS=$2; shift 2 ;;
    --wait)        WAIT_MIN=$2; shift 2 ;;
    --no-resource-limits) WORKER_ARGS+=(--no-resource-limits); shift ;;
    *) die "unknown flag $1 (see --help)" ;;
  esac
done
[ -n "$LUMI_RUN" ] || die "--lumi-run is required: the run dir that ./submit.sh printed on LUMI, or the training job's id"
[[ "$MAX_ENVS" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[0-9]+$ && "$WAIT_MIN" =~ ^[0-9]+$ ]] \
  || die "--max-envs, --num-workers and --wait take numbers"
say "checking docker"
timeout 60 docker info >/dev/null 2>&1 || die "docker not usable by $USER (is the daemon running, are you in the docker group?)"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))' || die "python3 >= 3.8 needed"
command -v setsid >/dev/null || die "setsid not found (util-linux)"
command -v curl >/dev/null || die "curl not found"

# BatchMode: fail instead of asking for a passphrase. ControlPath=none: test a fresh login, like the tunnel's.
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
say "checking ssh to $LUMI_HOST"
"${SSH[@]}" -o ControlPath=none "$LUMI_HOST" true \
  || die "cannot ssh to $LUMI_HOST without a prompt. Load your key into an agent first: eval \"\$(ssh-agent)\"; ssh-add ~/.ssh/<lumi key> (and accept the host key once with a plain ssh $LUMI_HOST)"

# Prints "<run dir> <job id> <job state, or GONE> <bridge URL, or ->". The URL holds the run's secret: it only
# travels over ssh and into the worker's environment, never onto a command line.
lookup() {
  "${SSH[@]}" "$LUMI_HOST" bash -s -- "$(printf '%q' "$LUMI_RUN")" <<'EOF'
run=$1
if [[ "$run" =~ ^[0-9]+$ ]]; then
  out=$(scontrol show job "$run" 2>/dev/null | grep -o 'StdOut=[^ ]*' | cut -d= -f2-)
  [ -n "$out" ] || { echo "job $run is not in the queue (pass the run dir instead)" >&2; exit 2; }
  run=$(dirname "$out")
fi
[ -f "$run/job.env" ] || { echo "$run on $(hostname): not a run dir (no job.env)" >&2; exit 2; }
grep -q '^export HARBOR_SANDBOXES=tilde$' "$run/job.env" \
  || { echo "$run: not a run with harbor_env.sandboxes: tilde" >&2; exit 2; }
run=$(cd "$run" && pwd -P)
id=$(cat "$run/jobid" 2>/dev/null || true)
state=""
if [ -n "$id" ]; then state=$(squeue -h -j "$id" -o %T 2>/dev/null || true); fi
echo "$run ${id:--} ${state:-GONE} $(cat "$run/bridge.url" 2>/dev/null || echo -)"
EOF
}

# --- wait for the job's bridge ---
say "looking up $LUMI_RUN on $LUMI_HOST"
SECONDS=0 LAST=""
while :; do
  INFO=$(lookup) || die "lookup on $LUMI_HOST failed (see above)"
  read -r RUN JOBID STATE URL <<< "$INFO"
  LUMI_RUN=$RUN   # a job id resolves only while Slurm remembers the job: look up the run dir from now on
  if [ "$STATE" = GONE ]; then
    die "LUMI job $JOBID is not in the queue: the run has ended (see $RUN/slurm-$JOBID.out on LUMI)"
  fi
  [ "$URL" = - ] || break
  if [ "$STATE $URL" != "$LAST" ]; then
    say "waiting for the bridge of LUMI job $JOBID ($STATE; up to $WAIT_MIN min)"
    LAST="$STATE $URL"
  fi
  (( SECONDS < WAIT_MIN * 60 )) || die "no bridge after $WAIT_MIN min (job $JOBID $STATE; see $RUN/slurm-$JOBID.out on LUMI)"
  sleep 30
done
NAME=$(basename "$RUN")
LOGDIR="$REPO/logs/tilde/$NAME"
mkdir -p "$LOGDIR"
HOSTPORT=${URL#http://}; HOSTPORT=${HOSTPORT%%/*}   # <compute node hsn0 IP>:<port>
TOKEN=${URL##*/}
# A free local port: 127.0.0.1 is shared by every user of this machine (they would still need the token).
LOCAL_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
LOCAL_URL="http://127.0.0.1:$LOCAL_PORT/$TOKEN"
say "LUMI run $RUN, job $JOBID: bridge at $HOSTPORT"

# setsid: Ctrl-C goes to this script and the worker, so the tunnel stays up while the worker cleans up.
# ServerAlive*: notice a dead link within a minute. ControlPath=none: our own connection, not a shared master.
open_tunnel() {
  setsid "${SSH[@]}" -N -o ControlPath=none -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
    -L "127.0.0.1:$LOCAL_PORT:$HOSTPORT" "$LUMI_HOST" 2>> "$LOGDIR/tunnel.log" &
  TUNNEL_PID=$!
  local _
  for _ in $(seq 30); do
    curl -sf --max-time 5 "$LOCAL_URL/status" >/dev/null && return 0
    kill -0 "$TUNNEL_PID" 2>/dev/null || break
    sleep 1
  done
  kill "$TUNNEL_PID" 2>/dev/null || true
  return 1
}

cleanup() {
  trap - EXIT
  if [ -n "${WORKER_PID:-}" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID" 2>/dev/null || true   # it removes the run's containers itself
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  if [ -n "${TUNNEL_PID:-}" ]; then kill "$TUNNEL_PID" 2>/dev/null || true; fi
  local ids
  ids=$(docker ps -aq --filter "label=hb-bridge.run=$NAME" 2>/dev/null || true)
  if [ -n "$ids" ]; then
    # shellcheck disable=SC2086
    docker rm -f $ids >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

open_tunnel || die "tunnel to $HOSTPORT via $LUMI_HOST failed, see $LOGDIR/tunnel.log ('administratively prohibited': the login node refuses forwarding; 'Connection refused': the bridge is not up)"
say "tunnel 127.0.0.1:$LOCAL_PORT -> $HOSTPORT via $LUMI_HOST (log $LOGDIR/tunnel.log)"

HB_BRIDGE_URL="$LOCAL_URL" python3 -u "$REPO/tilde/docker_worker.py" --name "tilde-$(hostname -s)" \
  --max-envs "$MAX_ENVS" --num-workers "$NUM_WORKERS" --run-label "$NAME" "${WORKER_ARGS[@]}" \
  > >(tee -a "$LOGDIR/worker.log") 2>&1 &
WORKER_PID=$!
say "worker running (log $LOGDIR/worker.log). Ctrl-C stops it and removes its containers."

# --- supervise: the worker, the tunnel, and (every minute) whether the LUMI job still runs ---
alive() {   # 0 while the job runs and its bridge is the one we connected to
  local info state url
  info=$(lookup) || return 2   # LUMI unreachable: no verdict
  read -r _ _ state url <<< "$info"
  [ "$state" != GONE ] && [ "$url" = "$URL" ]
}
TICK=0
while :; do
  sleep 10
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    rc=0; wait "$WORKER_PID" || rc=$?
    die "the worker exited (rc $rc), see $LOGDIR/worker.log"
  fi
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    say "tunnel closed (see $LOGDIR/tunnel.log), reopening"
    until open_tunnel; do
      rc=0; alive || rc=$?
      if (( rc == 1 )); then say "LUMI job $JOBID has ended: stopping"; exit 0; fi
      say "tunnel failed, retrying in 30 s"
      sleep 30
    done
    say "tunnel reopened"
  fi
  (( ++TICK % 6 == 0 )) || continue
  rc=0; alive || rc=$?
  if (( rc == 1 )); then say "LUMI job $JOBID has ended: stopping the worker"; exit 0; fi
  if (( rc == 2 )); then say "lookup on $LUMI_HOST failed; checking again in a minute"; fi
done
