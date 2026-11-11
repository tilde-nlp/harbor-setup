#!/bin/bash
#SBATCH --job-name=gpu-parity
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=checks/results/%x-%j.out
#
# vLLM vs transformers per-token log-probs on the model dir's golden prompts. One node, no Ray, no FSDP.
# `./check.sh gpu <run yaml>` prints the command with the right MODEL; by hand, from the repo root:
#     MODEL=$PWD/tildeopen sbatch checks/gpu_parity.sbatch
# The verdict is the last lines of the job output and is recorded for `./check.sh pre` (settings.gpu_parity_record).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
: "${MODEL:?set MODEL to the model directory (./check.sh gpu <run yaml> prints the command)}"
: "${SANDBOX:=$ROOT/multitorch_sandbox.sif}"
OUT="$ROOT/checks/results/gpu_parity-${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

run() {  # one python process per engine, so each one starts with empty GPUs
  singularity exec -B /var/spool/slurmd,/scratch/project_465003062,/pfs "$SANDBOX" bash -c "
    . /opt/venv/bin/activate
    export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_HOME=$ROOT/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
    export HIP_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}; unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    cd $ROOT && python -m checks.gpu_parity $*"
}

run vllm --model "$MODEL" --out "$OUT/vllm.npz"
run hf   --model "$MODEL" --out "$OUT/transformers.npz"
run compare "$OUT/vllm.npz" "$OUT/transformers.npz" --model "$MODEL" --root "$ROOT" --job "${SLURM_JOB_ID:-manual}"
