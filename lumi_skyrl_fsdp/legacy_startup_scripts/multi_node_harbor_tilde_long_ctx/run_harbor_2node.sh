set -x

# COLOCATED GRPO with Harbor (terminus-2) rollouts for TildeOpen-30b, multi-node LUMI / ROCm.
# 64k-context memory test: every task in $TASKS has a ~55k-token padded instruction.md, so each
# trajectory reaches ~60k tokens within 2 turns. Rewards will be 0; the point is to see whether
# rollout and the policy update fit at MAX_MODEL_LEN=65536.
#
# Launched by submit_2node.sbatch on the Ray head node, inside the sandbox, after skyrl_env.sh.
# Requires the harbor env job (submit_harbor_env.sbatch) to be running: APPTAINER_BRIDGE_URL is
# handed in by the sbatch from $ROOT/bridge.url.
#
# Layout: every node runs BOTH the FSDP policy and one vLLM engine, on the same GCDs.
# SkyRL time-shares them: the policy state is offloaded to host RAM while vLLM generates, and
# vLLM sleeps (weights to host, KV cache freed) while the policy trains. --mem=0 in the sbatch
# is required for the host-side copies.
#
#   NUM_NODES=2 -> 16 GCDs: FSDP policy, DP = 16 / SP = 4 at SP=4; vLLM TP=8 -> 2 engines.
#
#   training phase (vLLM asleep):
#     fp32 master + grads + Adam m/v = 16 B/param x 30.7B ~ 491 GB / 16 ~ 31 GB per GCD.
#     No ref model is built while use_kl_loss=false, so the ref_* placement flags are inert.
#     Activations (micro batch 1 + gradient checkpointing): 60 layers x seq x 6144 x 2 B
#       64k context: ~48 GB at SP=1, ~24 GB at SP=2, ~12 GB at SP=4
#     plus ~2 GB unsharded FSDP layers and ~4 GB recompute working set (16k tokens/rank at SP=4)
#     -> ~49 GB per GCD at 64k/SP=4. SP=2 would be ~63 GB and OOM.
#     Logits: logprobs_chunk_size=512 keeps them well under 1 GB.
#   rollout phase (policy offloaded):
#     30.7B bf16 at TP=8 = 7.7 GB weights per GCD; 0.80 x 64 GB -> ~43 GB KV cache per GCD,
#     ~31 KB/token/GCD (60 layers, 1 of 8 KV heads per rank, head dim 128) -> ~1.4M tokens/engine.
#     2 engines x 8 seqs x 64k = 1M tokens of demand, against 8 concurrent trials. Fits.
#
# Weight sync: with colocate_all=true SkyRL uses the intra-node CUDA-IPC handle path, not the
# cross-node RCCL broadcast of the split layout. TP=8 (one engine per node) is kept so every
# engine's workers live on the same node as the policy ranks that own their shards.
#
# If training OOMs: trainer.policy.fsdp_config.cpu_offload=true, or NUM_NODES=4 (then 15 GB of
# optimizer state per GCD and DP=8; MINI_BS*N_SAMPLES must stay a multiple of DP).

: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
: "${NUM_NODES:=${SLURM_JOB_NUM_NODES:-2}}"
: "${GPUS_PER_NODE:=8}"
: "${MODEL:=$ROOT/tildeopen}"        # pre-download into $HF_HOME: HF_HUB_OFFLINE=1 is set
: "${SERVED_NAME:=$(basename "$MODEL")}"   # no '/' allowed: Harbor addresses hosted_vllm/<name>
: "${TP:=8}"                        # vLLM tensor parallel per engine; must divide GPUS_PER_NODE
: "${SP:=4}"                        # Ulysses sequence parallel for policy/ref; 4 needed at 64k
: "${GPU_UTIL:=0.60}"               # vLLM share while awake; the policy is offloaded to host then
# Context: never above what the checkpoint supports (vLLM refuses max_model_len > max_position_embeddings;
# TildeOpen-30b config.json says 65536). Falls back to 32768 when $MODEL is a hub id rather than a directory.
MODEL_CTX=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["max_position_embeddings"])' "$MODEL" 2>/dev/null || echo 32768)
: "${MAX_MODEL_LEN:=$(( MODEL_CTX < 65536 ? MODEL_CTX : 65536 ))}"
: "${MAX_GEN_LEN:=2048}"            # cap per LLM call (one agent turn), not per trajectory
: "${MAX_TURNS:=2}"                 # 55k prompt + 2 x 2k stays under 64k without terminus-2 summarizing
: "${N_CONCURRENT:=8}"              # concurrent Harbor trials; <= total workers in the env job
: "${TASKS:="$ROOT/data/ctx64k"}"   # directory of Harbor task dirs with padded instruction.md
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
CHAT_TEMPLATE=/opt/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2

: "${APPTAINER_BRIDGE_URL:?not set - submit_2node.sbatch reads it from \$ROOT/bridge.url}"

# --- colocated: all nodes host both the policy and vLLM ---
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
NUM_ENGINES=$((NUM_GPUS / TP))      # rollout data parallel: independent vLLM replicas behind the router
DP=$((NUM_GPUS / SP))               # training data parallel: FSDP ranks / Ulysses group size

die() { echo "FATAL: $*" >&2; exit 1; }
[ "$NUM_NODES" -ge 1 ]              || die "NUM_NODES=$NUM_NODES"
[ $((GPUS_PER_NODE % TP)) -eq 0 ]   || die "TP=$TP must divide GPUS_PER_NODE=$GPUS_PER_NODE (one engine never spans nodes)"
[ $((NUM_GPUS % SP)) -eq 0 ]        || die "SP=$SP must divide NUM_GPUS=$NUM_GPUS"
[ "$DP" -ge 2 ]                     || die "training DP=$DP: no data parallelism left (NUM_GPUS=$NUM_GPUS SP=$SP) - lower SP or raise NUM_NODES"
[ $((NUM_ENGINES * TP)) -eq "$NUM_GPUS" ] || die "colocate_all needs NUM_ENGINES*TP == NUM_GPUS ($NUM_ENGINES*$TP != $NUM_GPUS)"
echo "layout: colocated ${NUM_NODES}n/${NUM_GPUS} GCDs | policy SP=$SP -> DP=$DP | vLLM TP=$TP -> $NUM_ENGINES engines | ctx $MAX_MODEL_LEN"

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
#   8 tasks x 4 samples = 32 trajectories/step; mini 8 -> 8*4/4 = 8 per DP rank at 2 nodes/SP=4,
#   4 per rank at 4 nodes. micro 1 -> ok either way. train_batch_size must be a multiple of mini.
TRAIN_BS=8
MINI_BS=8
N_SAMPLES=4
[ $(( MINI_BS * N_SAMPLES % DP )) -eq 0 ] || die "batch rule violated: $MINI_BS*$N_SAMPLES not divisible by DP=$DP"

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
  trainer.placement.colocate_all=true \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp \
  trainer.policy.sequence_parallel_size=$SP \
  trainer.ref.sequence_parallel_size=$SP \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.gradient_checkpointing=true \
  trainer.logprobs_chunk_size=512 \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.critic_num_nodes=$NUM_NODES \
  trainer.placement.critic_num_gpus_per_node=$GPUS_PER_NODE \
  generator.inference_engine.served_model_name="$SERVED_NAME" \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP \
  generator.inference_engine.distributed_executor_backend=ray \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE \
  generator.inference_engine.max_num_seqs=8 \
  generator.inference_engine.max_num_batched_tokens=16384 \
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
  trainer.project_name="harbor-ctx64k" \
  trainer.run_name="harbor_lumi_coloc_${NUM_NODES}n_${SERVED_NAME}_sp${SP}_dp${DP}" \
  trainer.resume_mode=null \
  trainer.log_path="$ROOT/logs" \
  trainer.ckpt_path="$ROOT/ckpts/harbor_${SERVED_NAME}_ckpt" \
  "$@"