#!/usr/bin/env bash
# Submit a run:   ./submit.sh configs/<task>/<run>.yaml [key.path=value ...]
# Check only:     ./submit.sh --dry-run configs/<task>/<run>.yaml [key.path=value ...]
#
# 1. launch/render.py resolves the config (+ command-line overrides), checks the layout and batch
#    rules, and writes a run dir: <paths.runs>/<timestamp>-<config name>/
# 2. this repo's launch/, configs/ and skyrl_extensions/ are copied into <run dir>/code (plus git state), so
#    the job runs exactly what was submitted
# 3. harbor configs: the environment plane job is submitted first, the training job after it
# 4. sbatch launch/job.sbatch <run dir>; Slurm output goes to <run dir>/slurm-<jobid>.out
#
# Examples:
#   ./submit.sh configs/single_step/gsm8k.yaml
#   ./submit.sh configs/benchmark/fullctx_3n.yaml env.NCCL_NET=Socket env.NCCL_DEBUG=WARN
#   ./submit.sh configs/pregen/grpo_bs5.yaml skyrl.trainer.resume_mode=latest slurm.time=06:00:00
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
die() { echo "submit.sh: $*" >&2; exit 1; }

DRY=()
if [ "${1:-}" = "--dry-run" ]; then DRY=(--dry-run); shift; fi
[ $# -ge 1 ] || die "usage: ./submit.sh [--dry-run] <config.yaml> [key.path=value ...]"

# render.py needs PyYAML: cray-python has it (system python3 on LUMI is 3.6 without it).
PY="${RENDER_PYTHON:-python3}"
if ! "$PY" -c 'import yaml' 2>/dev/null; then
  module load cray-python >/dev/null 2>&1 || true
  "$PY" -c 'import yaml' 2>/dev/null || die "no PyYAML for $PY: module load cray-python, or set RENDER_PYTHON"
fi

if [ ${#DRY[@]} -gt 0 ]; then exec "$PY" "$REPO/launch/render.py" "$@" --dry-run; fi

RUN_DIR=$("$PY" "$REPO/launch/render.py" "$@")
[ -d "$RUN_DIR" ] || die "render.py did not create a run dir"

# freeze the code that the job will run
mkdir -p "$RUN_DIR/code"
cp -r "$REPO/launch" "$REPO/configs" "$REPO/skyrl_extensions" "$RUN_DIR/code/"
find "$RUN_DIR/code" -name __pycache__ -prune -exec rm -rf {} +
{ echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
  echo "command: ./submit.sh $*"
  echo "--- git status --short"; git -C "$REPO" status --short 2>/dev/null || true; } > "$RUN_DIR/git.txt"
git -C "$REPO" diff HEAD > "$RUN_DIR/git.diff" 2>/dev/null || true

DEP=()
if [ -f "$RUN_DIR/harbor_env.sbatch_args" ]; then
  mapfile -t HARGS < "$RUN_DIR/harbor_env.sbatch_args"
  HJOB=$(sbatch --parsable "${HARGS[@]}" "$RUN_DIR/code/launch/harbor_env.sbatch" "$RUN_DIR")
  HJOB=${HJOB%%;*}
  echo "$HJOB" > "$RUN_DIR/harbor_env.jobid"
  DEP=(--dependency="after:$HJOB")
  echo "harbor env job: $HJOB"
fi

mapfile -t SARGS < "$RUN_DIR/sbatch_args"
JOB=$(sbatch --parsable "${DEP[@]}" "${SARGS[@]}" "$RUN_DIR/code/launch/job.sbatch" "$RUN_DIR") || {
  if [ -n "${HJOB:-}" ]; then scancel "$HJOB"; fi
  die "sbatch of the training job failed (harbor env job cancelled)"
}
JOB=${JOB%%;*}
echo "$JOB" > "$RUN_DIR/jobid"

LATEST="$(dirname "$RUN_DIR")/latest"
ln -sfn "$RUN_DIR" "$LATEST"
echo "training job: $JOB"
echo "run dir:      $RUN_DIR   (also $LATEST)"
echo "log:          $RUN_DIR/slurm-$JOB.out"
