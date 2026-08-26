#!/usr/bin/env bash
# Self-allocating launcher: salloc -> host staging -> container training.
set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/skyrl-harbor
SANDBOX=$ROOT/multitorch_sandbox
MODEL=Qwen/Qwen2.5-0.5B-Instruct

: "${ACCOUNT:=project_465003062}"
: "${PARTITION:=dev-g}"
: "${NODES:=1}"
: "${GPUS:=4}"
: "${TIME:=2:00:00}"

# ---------- phase 0: allocation ----------
if [ -z "${SLURM_JOB_ID:-}" ] || ! squeue -h -j "$SLURM_JOB_ID" >/dev/null 2>&1; then
  unset SLURM_JOB_ID SLURM_JOBID
  exec salloc --account="$ACCOUNT" --partition="$PARTITION" --nodes="$NODES" \
    --gpus-per-node="$GPUS" --ntasks="$NODES" --cpus-per-task=28 --mem=240G \
    --time="$TIME" bash "$0" "$@"
fi

# ---------- phase 1: host ----------
if [ "${SKYRL_STAGE:-host}" = host ]; then
  mkdir -p "$ROOT"/{data,hf,logs,skyrl-logs,ckpts,wandb} \
           "$ROOT"/tmp/{uv-cache,ray,scratch,miopen,skyrl} \
           "$SANDBOX/rt"

  # image is read-only under `singularity run`, so stage from the host side
  install -D -m 755 "$ROOT/run.sh" "$SANDBOX/opt/SkyRL/examples/train/gsm8k/run.sh"

  # compute nodes have no internet: fetch weights and dataset here
  HF_HOME=$ROOT/hf singularity exec --writable -B "$ROOT" "$SANDBOX" bash -c "
    . /opt/venv/bin/activate
    hf download $MODEL
    [ -f $ROOT/data/gsm8k/train.parquet ] || \
      python /opt/SkyRL/examples/train/gsm8k/gsm8k_dataset.py --output_dir $ROOT/data/gsm8k"

  export SINGULARITYENV_SKYRL_STAGE=container
  exec srun --overlap singularity run -B /var/spool/slurmd,"$ROOT","$ROOT/tmp/ray:/rt" \
    "$SANDBOX" bash "$0" "$@"
fi

# ---------- phase 2: container ----------
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# Ray on ROCm only reads HIP_VISIBLE_DEVICES; CUDA_* leaks in via the vLLM import.
# Do NOT set RAY_EXPERIMENTAL_NOSET_*: it breaks device assignment -> RCCL invalid usage.
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

# Everything lives under $ROOT. /rt is a short bind of $ROOT/tmp/ray because
# AF_UNIX socket paths cap at 107 bytes and $ROOT is already 62 chars.
export RAY_TMPDIR=/rt
export TMPDIR=$ROOT/tmp/scratch
export SKYRL_HOME=$ROOT/tmp/skyrl
export MIOPEN_USER_DB_PATH=$ROOT/tmp/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

export UV_PROJECT_ENVIRONMENT=/opt/venv
export UV_CACHE_DIR=$ROOT/tmp/uv-cache
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1

export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_GPU_METRICS=1 RAY_DEDUP_LOGS=0
unset RAY_LOG_TO_STDERR   # otherwise the autoscaler spams stderr

cd /opt/SkyRL
python -c "import torch; print('gcds:', torch.cuda.device_count())"

LOG=$ROOT/skyrl-logs/gsm8k-$(date +%F-%H%M%S).log
echo "ray session: $ROOT/tmp/ray/session_*  |  log: $LOG"
bash examples/train/gsm8k/run.sh "$@" 2>&1 | tee "$LOG"