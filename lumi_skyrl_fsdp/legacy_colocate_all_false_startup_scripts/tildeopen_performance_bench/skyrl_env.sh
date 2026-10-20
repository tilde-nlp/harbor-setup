# Sourced INSIDE the container by every Ray node (head + workers) and by the driver.
# Each raylet has its own environment and SkyRL/vLLM actors inherit it, so every
# variable an actor must see lives here, not in the sbatch or the driver.

: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
export ROOT

# --- Python / HF ---
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/pyhooks${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/hf"
export HF_HUB_OFFLINE=1
export HF_ENABLE_PARALLEL_LOADING=true     # rank 0 reads the safetensors shards concurrently
export HF_PARALLEL_LOADING_WORKERS=8

# --- GPU visibility ---
# Slurm sets ROCR_VISIBLE_DEVICES; Ray refuses it and wants HIP_VISIBLE_DEVICES.
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HWLOC_COMPONENTS=-amd_smi,-rsmi

# --- node-local caches (/tmp is per node, wiped at job end) ---
export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-${SLURM_JOB_ID:-x}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export TORCH_EXTENSIONS_DIR="/tmp/$USER-torch-ext-${SLURM_JOB_ID:-x}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$TORCH_EXTENSIONS_DIR"

# --- Ray ---
export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0
# symmetric-run defaults are 30 s; nodes here start staggered (checkpoint staging, container, venv).
export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=300
export RAY_py_gcs_connect_timeout_s=300

# --- SkyRL / vLLM ---
export SKYRL_LD_LIBRARY_PATH_EXPORT=True   # forward LD_LIBRARY_PATH to Ray workers
# Worker process-group timeout. Rank 0 alone loads the HF checkpoint (bf16 -> fp32 in host RAM)
# while the other ranks wait in the first gloo broadcast; TildeOpen-30b takes ~10 min from /scratch.
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=7

# =====================================================================
# Network: RCCL (aws-ofi plugin, libfabric/cxi), gloo
# =====================================================================
# Verified from per-process logs (2026-10-14): every rank loads aws-ofi-nccl 1.20.0 on
# libfabric 2.1, cxi provider, 4 NICs, DMA-BUF support, plugin name "Libfabric".

# Bootstrap/OOB traffic on the Slingshot NICs only.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0

# Require the plugin by its registered name. A missing plugin now aborts init instead of
# silently using TCP sockets. (Setting this to Socket forces sockets, which is what
# caused "via NET/Socket" in earlier runs.)
export NCCL_NET=Libfabric
# Some RCCL versions look for libnccl-net.so instead of librccl-net.so; pin the path when present.
[ -f /usr/lib/x86_64-linux-gnu/librccl-net.so ] && export NCCL_NET_PLUGIN=/usr/lib/x86_64-linux-gnu/librccl-net.so

# GPU Direct RDMA (NIC reads GPU memory directly; LUMI docs and HPE guide recommend PHB).
# PHB hung in rccl_test job 22206038 without the FI_* settings below; this is the re-test.
# If ncclCommInitRank hangs again, fall back to LOC (host-staged, ~58 GB/s for a 2-node all-reduce).
# Never set NCCL_DMABUF_ENABLE here: RCCL then reads /boot/config-*, which is not in the
# container, and segfaults in ncclCommInitRank (job 22817181). The plugin detects DMA-BUF itself.
export NCCL_NET_GDR_LEVEL=PHB

# libfabric/cxi provider settings (HPE RCCL tuning guide, LAIF image defaults).

# evil useless flags
# export FI_CXI_RDZV_THRESHOLD=0
# export FI_CXI_RDZV_GET_MIN=0
# export FI_CXI_RDZV_EAGER_SIZE=0
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export FI_CXI_DEFAULT_TX_SIZE=2048
# export FI_CXI_RX_MATCH_MODE=hybrid
# export NCCL_CROSS_NIC=1

export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export RCCL_MSCCL_ENABLE=0

# Weight-sync communicator (1 trainer GCD + 8 vLLM GCDs): the 8-GCD ranks match RCCL's MI250X
# topology model and build an extra connection set (log shows "Channel xx/2"), the 1-GCD rank
# builds set 0, and ncclCommInitRank deadlocks (job 22820955). Reproduced in
# scripts/rccl_hetero_test.sbatch. Under test: disable model matching on every rank and verify
# in the ENV lines of the per-process logs that all 9 ranks read it.
export RCCL_MODEL_MATCHING_DISABLE=1

# cxi IDC sends copy small messages out of the source buffer on the RCCL proxy thread via
# rocr_memcpy, which never completes while an RCCL kernel is resident on the GPU: every
# all-reduce over Libfabric hangs, broadcasts pass. Disabling IDC lets the NIC read the
# registered buffer directly. Reproducer: scripts/submit_rccl_test.sbatch; hang 22879416,
# pass 22896687 / 22897075 / 22897363 (2026-10-16). Report to LUMI support.
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
# without this flag - libfabric hangs on the shm_broadcast.py
# I still do not understand why this thing works, but it must somehow influence NCCL inner works, which solves the deadlock


# Debug: one RCCL log per process (%h host, %p pid). ENV prints the variables each rank read.
# Check after every multi-node run: no "Using network Socket" lines may appear.
# Lower NCCL_DEBUG to WARN once stable.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV,COLL
export NCCL_DEBUG_TIMESTAMP_LEVELS=ALL
export NCCL_DEBUG_FILE="$ROOT/logs/nccl-${SLURM_JOB_ID:-x}/%h.%p.log"
mkdir -p "$ROOT/logs/nccl-${SLURM_JOB_ID:-x}"