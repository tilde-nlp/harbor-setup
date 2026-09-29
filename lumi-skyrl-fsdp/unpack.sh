#!/usr/bin/env bash
set -euo pipefail
mkdir -p .
cat > harbor_worker_node.sh <<'EOF_d3031fb274ea'
#!/bin/bash
# Runs host-side on one node of the harbor env job (one srun task per node).
# Starts the worker.py that owns `singularity instance start` for this node.
# Expects ROOT, BRIDGE_URL, WORKERS_PER_NODE from the sbatch env.
set -euo pipefail
module load cray-python                       # host python3 is 3.6; worker needs >=3.7
APPT="$ROOT/marin-harbor/src/harbor/environments/apptainer"
export SINGULARITY_CACHEDIR="$ROOT/singularity_cache"
echo "[$SLURMD_NODENAME] worker x$WORKERS_PER_NODE -> $BRIDGE_URL"
exec python3 "$APPT/worker.py" --bridge-url "$BRIDGE_URL" \
     --sif-cache "$ROOT/sif_cache" --num-workers "$WORKERS_PER_NODE"
EOF_d3031fb274ea
mkdir -p .
cat > ray_node.sh <<'EOF_4de8111a9d04'
#!/bin/bash
# Runs on the HOST side as one task of the multi-node Ray step (one task per node). Starts
# exactly one Ray node inside the singularity sandbox and blocks for the lifetime of the job.
# Task 0 (SLURM_PROCID=0, the first node of the allocation = the batch host) is the head;
# every other task is a worker. An explicit role argument still works: bash ray_node.sh worker
# Expects ROOT, SANDBOX, SCRIPTS, RAY_PORT, RAY_ADDRESS, GPUS_PER_NODE from the sbatch env.
set -euo pipefail

ROLE="${1:-$([ "${SLURM_PROCID:-0}" = 0 ] && echo head || echo worker)}"

# This node's Slingshot address (same trick as for the head in the sbatch script).
MY_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)

NGPU=$(tr ',' '\n' <<< "${ROCR_VISIBLE_DEVICES:-}" | grep -c .)
[ "$NGPU" -eq "$GPUS_PER_NODE" ] || {
  echo "[$SLURMD_NODENAME] FATAL: step got $NGPU/$GPUS_PER_NODE GPUs (ROCR=$ROCR_VISIBLE_DEVICES)" >&2
  exit 1
}

# Slingshot credentials. Every Ray worker on this node (policy ranks, vLLM TP workers) inherits
# them from this raylet; without SLINGSHOT_VNIS the cxi provider cannot open a domain and RCCL
# ends up on TCP sockets (or, with NCCL_NET pinned, fails). Fail here, in the first minute.
echo "[$SLURMD_NODENAME] SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-<unset>} SLINGSHOT_DEVICES=${SLINGSHOT_DEVICES:-<unset>} SLINGSHOT_SVC_IDS=${SLINGSHOT_SVC_IDS:-<unset>}"
[ -n "${SLINGSHOT_VNIS:-}" ] || {
  echo "[$SLURMD_NODENAME] FATAL: no Slingshot VNI in this step - is this a single-node step? (see submit_2node.sbatch)" >&2
  exit 1
}

# Ray workers on this node inherit the environment of THIS raylet, not the driver's,
# which is why skyrl_env.sh is sourced here and not only in the driver step.
COMMON="--node-ip-address=$MY_IP --num-gpus=$GPUS_PER_NODE --num-cpus=${SLURM_CPUS_PER_TASK:-56} --block"

case "$ROLE" in
  head)
    RAY_CMD="ray start --head --port=$RAY_PORT --dashboard-host=0.0.0.0 --dashboard-port=8265 $COMMON"
    ;;
  worker)
    # Same step as the head, so wait until its GCS port answers before joining (up to ~3 min).
    HEAD_HOST=${RAY_ADDRESS%:*}
    for _ in $(seq 90); do (exec 3<>"/dev/tcp/$HEAD_HOST/$RAY_PORT") 2>/dev/null && break; sleep 2; done
    (exec 3<>"/dev/tcp/$HEAD_HOST/$RAY_PORT") 2>/dev/null || { echo "[$SLURMD_NODENAME] FATAL: Ray head $RAY_ADDRESS never came up" >&2; exit 1; }
    RAY_CMD="ray start --address=$RAY_ADDRESS $COMMON"
    ;;
  *)
    echo "unknown role: $ROLE" >&2; exit 1
    ;;
esac

echo "[$SLURMD_NODENAME] $ROLE @ $MY_IP: $RAY_CMD"
exec singularity exec -B "${BINDS:-/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038}" "$SANDBOX" \
  bash -c "source $SCRIPTS/skyrl_env.sh && ray stop --force >/dev/null 2>&1; $RAY_CMD"
EOF_4de8111a9d04
mkdir -p .
cat > run_harbor_2node.sh <<'EOF_5cb639e544f3'
set -x

# NON-colocated GRPO with Harbor (terminus-2) rollouts for TildeOpen-30b, multi-node LUMI / ROCm.
# Launched by submit_2node.sbatch on the Ray head node, inside the sandbox, after skyrl_env.sh.
# Requires the harbor env job (submit_harbor_env.sbatch) to be running: APPTAINER_BRIDGE_URL is
# handed in by the sbatch from $ROOT/bridge.url.
#
# Layout on 4 nodes x 8 GCDs (64 GB each): the cluster is split, nothing shares a GCD.
#
#   TRAIN_NODES=2 -> 16 GCDs: FSDP policy.       DP = 16 / SP = 8 at SP=2
#   INFER_NODES=2 -> 16 GCDs: vLLM, TP=4.        4 engines = 4 rollout replicas
#
#   training GCDs (dedicated: nothing is offloaded, nothing sleeps)
#     fp32 master + grads + Adam m/v = 16 B/param x 30.7B ~ 491 GB / 16 ~ 31 GB per GCD.
#     No ref model is built while use_kl_loss=false, so the ref_* placement flags are inert.
#     Activations (micro batch 1 + gradient checkpointing): 60 layers x seq x 6144 x 2 B
#       8k  context: ~6 GB at SP=1, ~3 GB at SP=2
#       32k context: ~24 GB at SP=1, ~12 GB at SP=2
#     plus ~2 GB unsharded FSDP layers and a few GB recompute working set -> ~40 GB at 8k/SP=2.
#     Logits: logprobs_chunk_size=512 keeps them well under 1 GB.
#   inference GCDs (dedicated: no sleep mode, so gpu_memory_utilization can be high)
#     30.7B bf16 at TP=4 = 15.4 GB weights per GCD; 0.85 x 64 GB -> ~35 GB KV cache per GCD,
#     ~61 KB/token/GCD (60 layers, 2 of 8 KV heads per rank, head dim 128) -> ~500k tokens/engine.
#     4 engines x 8 seqs = 32 slots, against 8 concurrent trials.
#
# Weight sync: with colocate_all=false, weight_sync_backend=nccl is a real RCCL broadcast
# (trainer rank 0 + all 16 vLLM workers, across nodes over Slingshot), NOT the CUDA-IPC path the
# colocated run used. First run: check the log for the broadcast group coming up on aws-ofi-rccl.
#
# If training OOMs: trainer.policy.fsdp_config.cpu_offload=true, or SP=4 (DP=4, batch rule holds),
# or TRAIN_NODES=3 (then DP=12 at SP=2 and MINI_BS*N_SAMPLES must become a multiple of 12).

: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
: "${NUM_NODES:=${SLURM_JOB_NUM_NODES:-2}}"
: "${GPUS_PER_NODE:=8}"
: "${MODEL:=$ROOT/tildeopen}"        # pre-download into $HF_HOME: HF_HUB_OFFLINE=1 is set
: "${SERVED_NAME:=$(basename "$MODEL")}"   # no '/' allowed: Harbor addresses hosted_vllm/<name>
: "${TRAIN_NODES:=2}"                # whole nodes for the FSDP policy; the rest run vLLM
: "${TP:=8}"                        # vLLM tensor parallel per engine; must divide GPUS_PER_NODE.
                                    # 8 = one engine per inference node, so the trainer<->vLLM RCCL
                                    # group has no intra-node edges between two engines' processes
                                    # (job 22211xxx hung in that group's connect phase with TP=4).
: "${SP:=2}"                        # Ulysses sequence parallel for policy/ref; 1, 2 or 4
: "${GPU_UTIL:=0.85}"               # vLLM owns its GCDs now, no need to leave room for the policy
# Context: never above what the checkpoint supports (vLLM refuses max_model_len > max_position_embeddings;
# TildeOpen-30b is 8192). Falls back to 32768 when $MODEL is a hub id rather than a directory.
MODEL_CTX=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["max_position_embeddings"])' "$MODEL" 2>/dev/null || echo 32768)
: "${MAX_MODEL_LEN:=$(( MODEL_CTX < 32768 ? MODEL_CTX : 32768 ))}"   # terminus-2 multi-turn context
: "${MAX_GEN_LEN:=4096}"            # cap per LLM call (one agent turn), not per trajectory
: "${MAX_TURNS:=12}"                # agent turns per trial; 12 x ~2k tokens stays well under 32k
: "${N_CONCURRENT:=8}"              # concurrent Harbor trials; <= total workers in the env job
: "${TASKS:="$ROOT/data/smoke"}"    # directory of Harbor task dirs (each with instruction.md)
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
CHAT_TEMPLATE=/opt/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2

: "${APPTAINER_BRIDGE_URL:?not set - submit_2node.sbatch reads it from \$ROOT/bridge.url}"

# --- split the cluster: TRAIN_NODES whole nodes for FSDP, the remaining nodes for vLLM ---
# Whole nodes, because the policy placement group asks Ray for {GPU: GPUS_PER_NODE} bundles: a
# node that vLLM has taken even one GCD from can no longer host a policy bundle.
INFER_NODES=$((NUM_NODES - TRAIN_NODES))
TRAIN_GPUS=$((TRAIN_NODES * GPUS_PER_NODE))
INFER_GPUS=$((INFER_NODES * GPUS_PER_NODE))
NUM_GPUS=$((TRAIN_GPUS + INFER_GPUS))
NUM_ENGINES=$((INFER_GPUS / TP))    # rollout data parallel: independent vLLM replicas behind the router
DP=$((TRAIN_GPUS / SP))             # training data parallel: FSDP ranks / Ulysses group size

die() { echo "FATAL: $*" >&2; exit 1; }
[ "$TRAIN_NODES" -ge 1 ] && [ "$INFER_NODES" -ge 1 ] || die "need >=1 train and >=1 inference node (NUM_NODES=$NUM_NODES TRAIN_NODES=$TRAIN_NODES)"
[ $((GPUS_PER_NODE % TP)) -eq 0 ]  || die "TP=$TP must divide GPUS_PER_NODE=$GPUS_PER_NODE (mp backend: one engine per node)"
[ $((TRAIN_GPUS % SP)) -eq 0 ]     || die "SP=$SP must divide TRAIN_GPUS=$TRAIN_GPUS"
[ "$DP" -ge 2 ]                    || die "training DP=$DP: no data parallelism left (TRAIN_GPUS=$TRAIN_GPUS SP=$SP) - lower SP or raise TRAIN_NODES"
[ "$NUM_ENGINES" -ge 2 ]           || die "rollout DP=$NUM_ENGINES engine(s) (INFER_GPUS=$INFER_GPUS TP=$TP) - lower TP or TRAIN_NODES"
echo "layout: train ${TRAIN_NODES}n/${TRAIN_GPUS} GCDs (SP=$SP -> DP=$DP) | infer ${INFER_NODES}n/${INFER_GPUS} GCDs (TP=$TP -> $NUM_ENGINES engines) | ctx $MAX_MODEL_LEN"

# --- wait until every node's GPUs have registered with the Ray head ---
python - <<'EOF'
import os, sys, time, ray
ray.init(address=os.environ["RAY_ADDRESS"])
want = int(os.environ["NUM_NODES"]) * int(os.environ["GPUS_PER_NODE"])
have = 0
for _ in range(60):
    have = ray.cluster_resources().get("GPU", 0)
    if have >= want:
        print(f"Ray cluster ready: {ray.cluster_resources()}")
        break
    time.sleep(5)
else:
    sys.exit(f"only {have}/{want} GPUs joined the Ray cluster")
EOF

# --- batch rule (skyrl/train/utils/utils.py): policy_mini_batch_size * n_samples_per_prompt / DP
# must be a positive integer divisible by micro_train_batch_size_per_gpu.
# DP here is the POLICY's: TRAIN_GPUS / SP, the inference GCDs do not count.
#   8 tasks x 4 samples = 32 trajectories/step; mini 8 -> 8*4/8 = 4 per DP rank at SP=2 (8 at SP=4),
#   micro 1 -> ok either way. train_batch_size must be a multiple of policy_mini_batch_size.
TRAIN_BS=8
MINI_BS=8
N_SAMPLES=4
[ $(( MINI_BS * N_SAMPLES % DP )) -eq 0 ] || { echo "batch rule violated: $MINI_BS*$N_SAMPLES not divisible by DP=$DP" >&2; exit 1; }

cd /opt/SkyRL   # examples.* imports resolve from the repo root

python -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data="['$TASKS']" \
  data.val_data="['$TASKS']" \
  harbor_trial_config.trials_dir="$ROOT/trials" \
  harbor_trial_config.environment.type=apptainer \
  harbor_trial_config.environment.kwargs.sif_cache="$ROOT/sif_cache" \
  harbor_trial_config.environment.override_memory_mb=2048 \
  harbor_trial_config.agent.name=terminus-2 \
  harbor_trial_config.agent.kwargs.max_turns=$MAX_TURNS \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens=$MAX_GEN_LEN \
  generator.rate_limit.enabled=true \
  generator.rate_limit.max_concurrency=$N_CONCURRENT \
  generator.rate_limit.trajectories_per_second=2 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp \
  trainer.policy.sequence_parallel_size=$SP \
  trainer.ref.sequence_parallel_size=$SP \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.gradient_checkpointing=true \
  trainer.logprobs_chunk_size=512 \
  trainer.placement.policy_num_nodes=$TRAIN_NODES \
  trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.ref_num_nodes=$TRAIN_NODES \
  trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.critic_num_nodes=$TRAIN_NODES \
  trainer.placement.critic_num_gpus_per_node=$GPUS_PER_NODE \
  generator.inference_engine.served_model_name="$SERVED_NAME" \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP \
  generator.inference_engine.distributed_executor_backend=ray \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE \
  generator.inference_engine.max_num_seqs=8 \
  generator.inference_engine.max_num_batched_tokens=8192 \
  generator.inference_engine.gpu_memory_utilization=$GPU_UTIL \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.batched=false \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.sampling_params.max_generate_length=$MAX_GEN_LEN \
  trainer.epochs=1 \
  trainer.train_batch_size=$TRAIN_BS \
  trainer.policy_mini_batch_size=$MINI_BS \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=$TRAIN_BS \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="harbor-smoke" \
  trainer.run_name="harbor_lumi_${TRAIN_NODES}t${INFER_NODES}i_${SERVED_NAME}_sp${SP}_dp${DP}" \
  trainer.resume_mode=null \
  trainer.log_path="$ROOT/logs" \
  trainer.ckpt_path="$ROOT/ckpts/harbor_${SERVED_NAME}_ckpt" \
  "$@"
EOF_5cb639e544f3
mkdir -p .
cat > skyrl_env.sh <<'EOF_b62b4d438629'
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
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# Pin the network. Without this RCCL silently fell back to TCP sockets ("via NET/Socket" in the
# infra log) in every run up to 2026-09-21, which is why a 123 GB fp32 state-dict broadcast took
# ~10 min. With NCCL_NET set, RCCL either uses the plugin or fails init with a WARN.
export NCCL_NET=Libfabric
# GPU Direct RDMA over cxi HANGS in this container (rccl_test job 22206038: 16 ranks start
# ncclCommInitRank, none complete, no WARN). Host-staged transfers (LOC) work at ~58 GB/s bus
# bandwidth for a 2-node all-reduce, vs single-digit GB/s over sockets. Do not raise this to
# 3/PHB without re-running scripts/rccl_test.sbatch (a v4 with NCCL_DMABUF_ENABLE=1 is the
# thing to try if GDR is ever needed).
export NCCL_NET_GDR_LEVEL=LOC
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
mkdir -p "$ROOT/logs/nccl-${SLURM_JOB_ID:-x}"

# --- harbor env plane: separate job (submit_harbor_env.sbatch), address handed in by the sbatch ---
export APPTAINER_BRIDGE_URL="${APPTAINER_BRIDGE_URL:?not set - submit_2node.sbatch reads it from \$ROOT/bridge.url}"
EOF_b62b4d438629
mkdir -p .
cat > submit_2node.sbatch <<'EOF_e6ef54b13d83'
#!/bin/bash
#SBATCH --job-name=skyrl-gsm8k-14b
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g        # dev-g works too if its node/time limits allow 2 nodes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0                       # whole node: colocated training offloads policy state to host RAM
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Multi-node SkyRL on LUMI-G.
#   sbatch $ROOT/scripts/submit_2node.sbatch
#
# Layout (all on shared scratch, visible from every node and bound into the sandbox):
#   $ROOT/scripts/submit_2node.sbatch   this file (runs on the host side, on the head node)
#   $ROOT/scripts/ray_node.sh           per-node helper: enters the container, starts one Ray node
#   $ROOT/scripts/skyrl_env.sh          env sourced INSIDE the container by every Ray node and the driver
#   $ROOT/scripts/run_gsm8k_2node.sh    the training script (runs inside the container on the head node)
#
# Change --nodes to 4 for Qwen3-32B; nothing else here needs editing, the training script
# derives NUM_GPUS from SLURM_JOB_NUM_NODES * GPUS_PER_NODE.
#
# NOTE on Slingshot VNIs (jobs 22207xxx/22210806, 2026-09-21): Slurm's hpe_slingshot plugin
# hands a step its network VNI and the SLINGSHOT_VNIS/SLINGSHOT_SVC_IDS environment only for
# MULTI-NODE steps. The old layout ran one single-node srun step per Ray node, so no Ray
# process ever had Slingshot credentials: libfabric's cxi provider failed fi_domain() with
# ENOSYS ("Couldn't create domain. RC: -38") and RCCL silently fell back to TCP sockets.
# --network=job_vni did not change that here (site policy). So all Ray nodes now run as ONE
# plain multi-node step: task 0 is the head, the rest are workers (ray_node.sh picks the role
# from SLURM_PROCID). One step = one VNI shared by every rank, which is also what lets the
# trainer's rank 0 and the vLLM workers on other nodes form an RCCL group.
#
# NOTE on step layout: no srun step here uses --overlap, deliberately. Slurm 25.05
# (on LUMI since the 2026-09-02 maintenance) allocates only 7 of 8 GCDs to any
# --overlap step that covers a subset of a multi-node job, and blocks the 8th in
# the step's device cgroup (fixed upstream only in 26.05.3). Plain steps get all 8.
# So the ray nodes run as plain steps (one per node, never sharing a node), and the
# driver runs directly in this batch step instead of its own srun step.

set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export SANDBOX="$ROOT/multitorch_sandbox"
export SCRIPTS="$ROOT/scripts"
export GPUS_PER_NODE=8
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RAY_PORT=6379

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"

# The batch script runs on the first node of the allocation, which is also our Ray
# head node -- assert that so the local hsn0 lookup below can't silently be wrong.
[ "${SLURMD_NODENAME:-$(hostname)}" = "$HEAD_NODE" ] || {
  echo "FATAL: batch host ${SLURMD_NODENAME:-$(hostname)} != first node $HEAD_NODE" >&2
  exit 1
}

# Ray and RCCL traffic must stay on the Slingshot fabric (hsn*), not the management
# network. We're on the head node, so read its hsn0 address directly (no srun needed).
HEAD_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
echo "nodes: ${NODES[*]}   head: $HEAD_NODE ($RAY_ADDRESS)"

BRIDGE_FILE=$ROOT/bridge.url
for _ in $(seq 60); do [ -s "$BRIDGE_FILE" ] && break; sleep 5; done   # up to 5 min
[ -s "$BRIDGE_FILE" ] || { echo "FATAL: no $BRIDGE_FILE - submit_harbor_env.sbatch not running?" >&2; exit 1; }
export APPTAINER_BRIDGE_URL=$(cat "$BRIDGE_FILE")
curl -sf --max-time 5 "$APPTAINER_BRIDGE_URL/status" \
  || { echo "FATAL: bridge $APPTAINER_BRIDGE_URL unreachable from $SLURMD_NODENAME" >&2; exit 1; }
echo "harbor bridge: $APPTAINER_BRIDGE_URL"



# 1+2. One plain multi-node step for all Ray nodes (see the VNI note above). Each task owns
#      its whole node: 8 GCDs, 56 cores. ray_node.sh asserts the GPU count and the VNI, starts
#      the head on task 0 and a worker (which waits for the head) everywhere else, and blocks.
srun --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
     --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     bash "$SCRIPTS/ray_node.sh" &
RAY_STEP_PID=$!
sleep 30
kill -0 "$RAY_STEP_PID" 2>/dev/null || { echo "FATAL: the Ray step exited during startup (see the [nidXXXXXX] lines above)" >&2; exit 1; }

# 3. Driver: runs here, inside the batch step (which holds all 8 GCDs and all
#    cores on this node -- no srun, no --overlap). The driver only talks to the
#    Ray head over TCP; GPU work happens in actors under the raylets, and those
#    inherit the raylet's environment, not the driver's.
#    run_harbor_2node.sh first waits until all GPUs have joined the cluster.
export BINDS="/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038"
singularity exec -B "$BINDS" "$SANDBOX" \
  bash -c "source $SCRIPTS/skyrl_env.sh && bash $SCRIPTS/run_harbor_2node.sh"

# When the driver returns, this script exits and Slurm tears down the Ray step on all
# nodes, so nothing survives into the next job.
EOF_e6ef54b13d83
mkdir -p .
cat > submit_harbor_env.sbatch <<'EOF_0460446de233'
#!/bin/bash
#SBATCH --job-name=harbor-env
#SBATCH --account=project_465003062
#SBATCH --partition=small           # smoke: 1 shared node. Scale: --partition=standard, whole nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/logs/harbor-env-%j.out
set -euo pipefail

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export SCRIPTS=$ROOT/scripts
export WORKERS_PER_NODE=${WORKERS_PER_NODE:-8}   # total workers >= SkyRL's max_concurrency
export BRIDGE_PORT=${BRIDGE_PORT:-9910}
BRIDGE_FILE=$ROOT/bridge.url
mkdir -p $ROOT/sif_cache $ROOT/singularity_cache $ROOT/logs

# 1. bridge, host side of node 0 (this batch host). Advertise the Slingshot address.
BRIDGE_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
export BRIDGE_URL="http://$BRIDGE_IP:$BRIDGE_PORT"
( module load cray-python && exec python3 \
    $ROOT/marin-harbor/src/harbor/environments/apptainer/server.py --port $BRIDGE_PORT ) \
    > $ROOT/logs/bridge-$SLURM_JOB_ID.log 2>&1 &
sleep 3
curl -sf "$BRIDGE_URL/status" >/dev/null || { echo "FATAL: bridge failed to start" >&2; exit 1; }

# 2. publish the address; withdraw it whenever this job ends for any reason.
echo "$BRIDGE_URL" > "$BRIDGE_FILE"
trap 'rm -f "$BRIDGE_FILE"' EXIT
echo "bridge $BRIDGE_URL (job $SLURM_JOB_ID) published to $BRIDGE_FILE"

# 3. one worker process per node (node 0 included), blocks for the job lifetime.
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK --export=ALL \
     bash $SCRIPTS/harbor_worker_node.sh
EOF_0460446de233
