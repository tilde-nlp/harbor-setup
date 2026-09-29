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