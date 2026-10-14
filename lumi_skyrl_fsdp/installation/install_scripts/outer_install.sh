#!/usr/bin/env bash
#SBATCH --account=project_465003062
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=sandbox
#SBATCH --output=sandbox-%j.log
#
# install.sh — sandbox from the laifs multitorch-plus container. Stops before
# any package install; do that by hand inside, then write it down in INSTALL.md.
#
#   cd /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
#   sbatch install.sh
#   tail -f sandbox-<jobid>.log
set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif
ROOT="$SLURM_SUBMIT_DIR"
SANDBOX="$ROOT/multitorch_sandbox"

ln -sfn "$SIF" "$ROOT/multitorch_plus.sif"

if [ ! -d "$SANDBOX" ]; then
  singularity build --sandbox "$SANDBOX.partial" "$SIF"
  mv "$SANDBOX.partial" "$SANDBOX"
fi

mkdir -p "$SANDBOX"/{users,project,projappl,flash,scratch,appl,pfs,var/spool/slurmd}
mkdir -p "$ROOT"/{hf,logs,wandb,data,cache/torch_extensions}

echo "sandbox ready: $SANDBOX"
echo "enter:  srun --account=project_465003062 --partition=small -N1 -n1 -c16 --mem=64G -t 4:00:00 --pty \\"
echo "          singularity shell --writable $SANDBOX"
echo "inside: . /opt/venv/bin/activate; export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1"