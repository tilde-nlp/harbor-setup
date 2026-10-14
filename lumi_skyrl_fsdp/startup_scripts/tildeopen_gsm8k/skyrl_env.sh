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

# RAY CORES USED FOR WEIGHT SYNC
export OMP_NUM_THREADS=7

# --- torch.distributed timeout in the policy/ref workers ---
# SkyRL inits the worker process group with timeout=SKYRL_WORKER_NCCL_TIMEOUT_IN_S (default 600 s).
# At init only rank 0 reads the HF checkpoint (bf16 -> fp32 in host RAM); every other rank is
# already blocked in the first gloo broadcast of fsdp2_load_full_state_dict and dies with
# "Timed out waiting 600000ms for recv" if rank 0 needs longer than that. TildeOpen-30b (61 GB)
# took ~10.5 min from /scratch, so 600 s is not enough. This is read in the actor process, so it
# has to live here (raylet env), not in the driver.
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600
# Let transformers read the 13 safetensors shards concurrently instead of one tensor at a time
# (ignored by transformers < 4.53). Rank 0 is the only reader, so this is the cheap speed-up.
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=8
# Pin gloo (the CPU side of the "cpu:gloo,cuda:nccl" group) to Slingshot like RCCL above.
export GLOO_SOCKET_IFNAME=hsn0

# --- RCCL over Slingshot ---
# The single-node NCCL_NET_PLUGIN=none and NCCL_SOCKET_IFNAME=hsn0 lines are gone on purpose:
# cross-node collectives (FSDP all-gathers/reduce-scatters, trainer->vLLM weight sync) need the
# aws-ofi (Libfabric/cxi) plugin and all four hsn interfaces.
export NCCL_SOCKET_IFNAME=hsn0
# Pin the network. Without this RCCL silently fell back to TCP sockets ("via NET/Socket" in the
# infra log) in every run up to 2026-09-21, which is why a 123 GB fp32 state-dict broadcast took
# ~10 min. With NCCL_NET set, RCCL either uses the plugin or fails init with a WARN.
export NCCL_NET=Socket
# GPU Direct RDMA over cxi HANGS in this container (rccl_test job 22206038: 16 ranks start
# ncclCommInitRank, none complete, no WARN). Host-staged transfers (LOC) work at ~58 GB/s bus
# bandwidth for a 2-node all-reduce, vs single-digit GB/s over sockets. Do not raise this to
# 3/PHB without re-running scripts/rccl_test.sbatch (a v4 with NCCL_DMABUF_ENABLE=1 is the
# thing to try if GDR is ever needed).
export NCCL_NET_GDR_LEVEL=LOC # LOOK AT gpt-neox - THEY USE PHB. THIS FLAG SETS DIRECT MEMORY ACCESS FOR GPUS
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
# Verify on every multi-node run:  grep -c "via NET/Libfabric" $SKYRL_LOG_FILE  must be > 0 and
# grep -c "via NET/Socket" must be 0. Turn NCCL_DEBUG down to WARN once the setup is stable.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS=INIT,NET
# One RCCL log per process, written by RCCL itself. Through Ray's stdout capture the NCCL lines
# of 50+ processes arrive interleaved, block-buffered and truncated at kill time, which made the
# last two hangs unreadable. %h = host, %p = pid.
export NCCL_DEBUG_FILE="$ROOT/logs/nccl-${SLURM_JOB_ID:-x}/%h.%p.log"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Weight sync is one RCCL communicator of 1 trainer GCD + 8 GCDs per vLLM node. RCCL's Rome
# topology-model matching is a per-process decision (needs all 8 GCDs of the node in the comm):
# the 8-GCD ranks match MI250X model 36 and build an extra ring connection set, the 1-GCD rank
# does not, and ncclCommInitRank deadlocks after "Connected all rings". Reproduced in
# scripts/rccl_hetero_test.sbatch (jobs 22551317 hang / 22551365 ok); training jobs 22231718,
# 22429724, 22494530, 22526638, 22546678, 22549832 all died this way. Must be set on every rank.
# export RCCL_MODEL_MATCHING_DISABLE=1 - THIS DOESN'T WORK

mkdir -p "$ROOT/logs/nccl-${SLURM_JOB_ID:-x}"