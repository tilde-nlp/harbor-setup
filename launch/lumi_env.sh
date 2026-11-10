# LUMI environment for SkyRL inside the container. Sourced by launch/job.sbatch on every Ray node
# (role `inner`) before `ray start`; the driver and every SkyRL/vLLM actor inherit it from the
# raylet, so every variable an actor must see lives here. The config's `env:` section is applied
# after this file and wins (e.g. env: {NCCL_NET: Socket, NCCL_DEBUG: WARN}).
#
# Needs from job.env: RUN_DIR, CODE_DIR, HF_HOME.
#
#   1. common settings (always on)
#   2. transport: Libfabric (default)
#   3. transport: Socket
#   4. debug / logging
# shellcheck shell=bash

# ------------------------------------------------ 1. COMMON
# --- Python / HF ---
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"   # skyrl_extensions/ from the run's code snapshot
export HF_HOME
export HF_HUB_OFFLINE=1
export HF_ENABLE_PARALLEL_LOADING=true     # rank 0 reads the safetensors shards concurrently
export HF_PARALLEL_LOADING_WORKERS=8

# --- GPU visibility ---
# Slurm sets ROCR_VISIBLE_DEVICES; Ray refuses it and wants HIP_VISIBLE_DEVICES.
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HWLOC_COMPONENTS=-amd_smi,-rsmi

# --- node-local caches (/tmp is RAM, per node, charged to the job's memory cgroup) ---
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

# --- network settings shared by both transports ---
# Bootstrap/OOB traffic on the Slingshot NICs. Under Socket this list is also the data path,
# so keep it identical across benchmark runs.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export RCCL_MSCCL_ENABLE=0
# Weight-sync communicator is 1 trainer GCD + 8 GCDs per vLLM node. Without this the 8-GCD ranks
# match RCCL's MI250X topology model and build an extra connection set, the 1-GCD rank does not,
# and ncclCommInitRank deadlocks (job 22820955). Must be set on every rank. See notes/BUGS.md.
export RCCL_MODEL_MATCHING_DISABLE=1

# ------------------------------------------------ 2. TRANSPORT: LIBFABRIC (default)
# (aws-ofi-nccl 1.20.0 plugin on libfabric 2.1, cxi provider)
export NCCL_NET=Libfabric
# Some RCCL versions look for libnccl-net.so instead of librccl-net.so; pin the path when present.
if [ -f /usr/lib/x86_64-linux-gnu/librccl-net.so ]; then
  export NCCL_NET_PLUGIN=/usr/lib/x86_64-linux-gnu/librccl-net.so
fi
# GPU Direct RDMA level (LUMI docs and HPE guide recommend PHB). LOC = host-staged.
# Never set NCCL_DMABUF_ENABLE here: RCCL then reads /boot/config-*, which is not in the
# container, and segfaults in ncclCommInitRank (job 22817181). The plugin detects DMA-BUF itself.
export NCCL_NET_GDR_LEVEL=PHB
# libfabric/cxi provider settings (HPE RCCL tuning guide, LAIF image defaults).
# kdreg2 is requested but this container's libfabric cannot use it ("kdreg2 monitor not
# available" with FI_LOG_LEVEL=warn); libfabric falls back to another monitor. Harmless, noted.
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
# REQUIRED. Without it every all-reduce over Libfabric hangs while broadcasts pass (weight sync
# hangs in PyNcclCommunicator.__init__, vLLM then reports shm_broadcast.py "No available shared
# memory broadcast block"). Cause, from the stack of the stuck rank (job 22895140): for a small
# message the cxi provider builds an immediate-data command (IDC) by copying the bytes out of the
# source buffer; that copy runs through rocr_memcpy on the RCCL proxy thread and waits on the GPU,
# while the RCCL kernel on that GPU waits for the send. Circular wait, no error. With IDC disabled
# the NIC reads the registered buffer directly and no copy is made. Reported to LUMI support.
# Reproducer (hang 22879416; pass 22896687, 22897075, 22897363), kept at the pre-restructure tag:
#   git show pre-restructure:lumi_skyrl_fsdp/legacy_colocate_all_false_startup_scripts/rccl_bug_repro/submit_rccl_test.sbatch
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1

# ------------------------------------------------ 3. TRANSPORT: SOCKET
# (TCP over the hsn interfaces listed in NCCL_SOCKET_IFNAME)
# Everything in section 2 is inert under Socket (plugin pin, GDR level, FI_* variables), so the
# only variable that has to change is NCCL_NET. Per run, in the config:  env: {NCCL_NET: Socket}
# or on the command line:  ./submit.sh <config> env.NCCL_NET=Socket

# ------------------------------------------------ 4. DEBUG / LOGGING
# One RCCL log per process (%h host, %p pid) in the run dir. ENV prints the variables each rank read.
# Check after every multi-node run: "Using network Libfabric" or "Using network Socket" as intended.
# For benchmark runs use NCCL_DEBUG=WARN or drop COLL: at INFO, COLL logs every collective to Lustre.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV,COLL
export NCCL_DEBUG_TIMESTAMP_LEVELS=ALL
export NCCL_DEBUG_FILE="$RUN_DIR/nccl/%h.%p.log"
mkdir -p "$RUN_DIR/nccl"
