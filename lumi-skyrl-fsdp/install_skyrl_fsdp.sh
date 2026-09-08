#!/usr/bin/env bash
# install_skyrl_fsdp_lock.sh — SkyRL (FSDP) into a fresh laifs-plus sandbox.
# Replays skyrl.lock with pip (--no-deps: satisfied pins incl. +lumi builds
# are skipped, the rest install at the locked version), then the editables.
#
# To generate a lock file from a working setup:
# singularity exec /path/to/working_sandbox bash -c \
#   '. /opt/venv/bin/activate; pip freeze --exclude-editable' > skyrl.lock
#
#   cd /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
#   ./install_skyrl_fsdp_lock.sh [path/to/skyrl.lock]    # default: ./skyrl.lock
set -euo pipefail

ROOT="$(pwd)"
LOCK="${1:-$ROOT/skyrl.lock}"
ACCOUNT="${ACCOUNT:-project_465003062}"
SKYRL_VERSION="${SKYRL_VERSION:-0.3.0}"
SANDBOX="$ROOT/multitorch_sandbox"
SIF="/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif"

[ -f "$LOCK" ] || { echo "ERROR: lock file not found: $LOCK"; exit 1; }

run_compute() {
  if command -v srun >/dev/null 2>&1 && [ -z "${NO_SRUN:-}" ]; then
    srun --account="$ACCOUNT" --partition=small -N1 -n1 -c16 --mem=64G -t 2:00:00 "$@"
  else
    "$@"
  fi
}

#--- container + sandbox -------------------------------------------------------
ln -sfn "$SIF" multitorch_plus.sif
if [ ! -d "$SANDBOX" ]; then
  echo "==> building sandbox..."
  run_compute singularity build --sandbox "$SANDBOX" multitorch_plus.sif
fi

#--- dirs ----------------------------------------------------------------------
( cd "$SANDBOX" && mkdir -p users project projappl flash scratch appl pfs )
mkdir -p "$ROOT"/hf "$ROOT"/logs "$ROOT"/wandb "$ROOT"/cache/torch_extensions "$ROOT"/data

#--- SkyRL release -------------------------------------------------------------
if [ ! -d "$ROOT/SkyRL" ]; then
  git clone --depth 1 --branch "skyrl-v${SKYRL_VERSION}" \
    https://github.com/NovaSky-AI/SkyRL.git "$ROOT/SkyRL"
  rm -rf "$ROOT/SkyRL/.git"
fi
mkdir -p "$SANDBOX/opt/SkyRL"
cp -a "$ROOT/SkyRL/." "$SANDBOX/opt/SkyRL/"
tr -d '\r' < "$LOCK" | grep -Ev '^\s*(#|$)| @ ' > "$SANDBOX/opt/skyrl.lock"

#--- in-container install ------------------------------------------------------
cat > "$SANDBOX/opt/inner_install.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

pip install --no-deps -r /opt/skyrl.lock

cd /opt/SkyRL
cp docker/pyproject.amd.toml pyproject.toml
pip install --no-deps -e /opt/SkyRL/skyrl-gym
pip install --no-deps -e /opt/SkyRL/skyrl-train
pip install --no-deps -e /opt/SkyRL

# vLLM reuse_port source fix (delete if unwanted)
python - <<'PYEOF'
from pathlib import Path
p = Path("/opt/SkyRL/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py")
s = p.read_text() if p.exists() else ""
old = "    sock = create_server_socket(sock_addr, reuse_port=False)"
new = """    try:
        sock = create_server_socket(sock_addr, reuse_port=False)
    except TypeError:
        sock = create_server_socket(sock_addr)"""
if old in s: p.write_text(s.replace(old, new, 1)); print("patched")
else: print("patch skipped")
PYEOF

python -c "import torch, vllm, ray, skyrl; \
from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend; \
print('ok', torch.__version__, vllm.__version__, ray.__version__)"
pip check || echo "note: pip check complaints (may pre-exist in base)"
INNER
chmod +x "$SANDBOX/opt/inner_install.sh"

run_compute singularity exec --writable "$SANDBOX" bash /opt/inner_install.sh
rm -f "$SANDBOX/opt/inner_install.sh"
echo "Done: SkyRL $SKYRL_VERSION [fsdp] in $SANDBOX"