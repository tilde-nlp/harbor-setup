# Sourced INSIDE the container by every Ray node (head + workers) and by the driver.
#
# In the single-node script the driver shell and the raylet were the same shell, so exporting
# things before `ray start` was enough. With several nodes each raylet has its own environment,
# and SkyRL/vLLM workers inherit *that* one, so everything RCCL/HIP/HF related lives here.

: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
export ROOT

# --- container env ---
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/pyhooks${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/hf"
export HF_HUB_OFFLINE=1
# Ray workers do not inherit LD_LIBRARY_PATH (ROCm + libfabric paths); SkyRL forwards it when set.
export SKYRL_LD_LIBRARY_PATH_EXPORT=True

# --- GPU visibility ---
# Slurm hands each task ROCR_VISIBLE_DEVICES; Ray refuses that and wants HIP_VISIBLE_DEVICES.
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HWLOC_COMPONENTS=-amd_smi,-rsmi

# --- node-local scratch (/tmp is per node, wiped at job end) ---
export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-${SLURM_JOB_ID:-x}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export TORCH_EXTENSIONS_DIR="/tmp/$USER-torch-ext-${SLURM_JOB_ID:-x}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$TORCH_EXTENSIONS_DIR"

export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0

# --- RCCL over Slingshot (standard LUMI settings) ---
# The single-node NCCL_NET_PLUGIN=none and NCCL_SOCKET_IFNAME=hsn0 lines are gone on purpose:
# cross-node collectives (FSDP all-gathers/reduce-scatters) need the aws-ofi-rccl plugin and all
# four hsn interfaces. Without NCCL_SOCKET_IFNAME RCCL picks the management interface and hangs.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
# First multi-node run: NCCL_DEBUG=INFO and grep the infra log for "NET/OFI Using aws-ofi-rccl".
# If you see "NET/Socket" instead, librccl-net.so is not on LD_LIBRARY_PATH inside the sandbox.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS=INIT,NET
