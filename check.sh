#!/usr/bin/env bash
# The check suite for this setup: catches a run that cannot be trusted before GPU hours are spent on it
# or conclusions are drawn from it. What each check is for: notes/TRAINING_SANITY_CHECK.md.
#
#   ./check.sh pre    <run.yaml> [key.path=value ...]   before submitting; ./submit.sh runs this itself
#   ./check.sh gpu    <run.yaml> [key.path=value ...]   prints the sbatch command of the 1-node job that compares
#                                                       vLLM and transformers log-probs. Once per new model dir /
#                                                       container; `pre` shows its result afterwards
#   ./check.sh run    <run dir | slurm log> [--table N] after step 1 of a run, and at its end
#   ./check.sh export <export dir> --base <model dir>   before evaluating an export
#   ./check.sh list                                     every check, by stage
#   ./check.sh selftest                                 do the checks still catch the bugs they were written for?
#
# Runs on a login node (no GPU) with the container's Python. Exit code 1 if a check FAILs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SANDBOX="${SANDBOX:-$ROOT/multitorch_sandbox.sif}"
BINDS="${CHECK_BINDS:-/scratch/project_465003062,/pfs}"
die() { echo "check.sh: $*" >&2; exit 1; }
[ $# -ge 1 ] || { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
[ -e "$SANDBOX" ] || die "container $SANDBOX missing (a fresh clone needs ./setup.sh first)"
STAGE=$1; shift

# quiet libraries: the checks print their own verdicts
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error VLLM_LOGGING_LEVEL=ERROR HF_HOME="${HF_HOME:-$ROOT/hf}"
cd "$ROOT"

case "$STAGE" in
  pre|run|export|list)
    singularity exec -B "$BINDS" "$SANDBOX" /opt/venv/bin/python -m checks.runner "$STAGE" "$@"
    ;;
  selftest)
    singularity exec -B "$BINDS" "$SANDBOX" /opt/venv/bin/python -m pytest -q -p no:cacheprovider checks/tests "$@"
    ;;
  gpu)
    MODEL=$(singularity exec -B "$BINDS" "$SANDBOX" /opt/venv/bin/python -m checks.runner model-dir "$@") \
      || die "cannot resolve the model directory of $*"
    echo "submit from $ROOT:"
    echo "  MODEL=$MODEL sbatch checks/gpu_parity.sbatch"
    echo "when it has finished, ./check.sh pre $* shows the result (settings.gpu_parity_record)"
    ;;
  *)
    die "unknown stage '$STAGE' (pre, gpu, run, export, list, selftest)"
    ;;
esac
