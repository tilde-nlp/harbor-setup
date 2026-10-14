set -x

# Colocated GRPO with Harbor (terminus-2) rollouts for Qwen3-14B, multi-node LUMI / ROCm.
# Launched by submit_2node.sbatch on the Ray head node, inside the sandbox, after skyrl_env.sh.
# Requires the harbor env job (submit_harbor_env.sbatch) to be running: APPTAINER_BRIDGE_URL is
# handed in by the sbatch from $ROOT/bridge.url.
#
# Sized for 32k-token trajectories on 2 nodes x 8 GCDs (64 GB each), everything conservative:
#
#   training phase (vLLM asleep, its weights offloaded to host RAM)
#     FSDP over 16 GCDs: fp32 master + Adam m/v + bf16 params + grads ~ 300 GB / 16 ~ 19 GB per GCD,
#     plus ~2 GB for the sharded bf16 ref model.
#     Activations: micro batch 1 + gradient checkpointing ~ 20 GB per GCD for one 32k sequence at
#     SP=1. Ulysses sequence parallel SP=2 splits every sequence over 2 GCDs -> ~10 GB, ~30 GB total.
#     Logits: 32k x 152k vocab in fp32 would be ~20 GB; logprobs_chunk_size=512 caps it at ~0.3 GB.
#   generation phase (policy/ref/optimizer offloaded, vLLM awake)
#     14B bf16 at TP=2 = 14.8 GB weights per GCD; gpu_memory_utilization 0.60 -> ~24 GB KV cache
#     ~ 290k tokens ~ 9 full 32k sequences per engine. max_num_seqs=8 keeps vLLM inside that.
#     8 engines x 8 seqs = 64 slots, against 8 concurrent trials.
#
# If training still OOMs: SP=4 (dp=4; the batch rule below still holds), then
# trainer.policy.fsdp_config.cpu_offload=true (2 x 512 GB host RAM is plenty).

: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
: "${NUM_NODES:=${SLURM_JOB_NUM_NODES:-2}}"
: "${GPUS_PER_NODE:=8}"
: "${MODEL:=Qwen/Qwen3-14B}"        # pre-download into $HF_HOME: HF_HUB_OFFLINE=1 is set
: "${SERVED_NAME:=$(basename "$MODEL")}"   # no '/' allowed: Harbor addresses hosted_vllm/<name>
: "${TP:=2}"                        # vLLM tensor parallel per engine; must divide GPUS_PER_NODE
: "${SP:=2}"                        # Ulysses sequence parallel for policy/ref; 1, 2 or 4
: "${MAX_MODEL_LEN:=32768}"         # terminus-2 multi-turn context; matches agent model_info
: "${MAX_GEN_LEN:=4096}"            # cap per LLM call (one agent turn), not per trajectory
: "${MAX_TURNS:=12}"                # agent turns per trial; 12 x ~2k tokens stays well under 32k
: "${N_CONCURRENT:=8}"              # concurrent Harbor trials; <= total workers in the env job
: "${TASKS:="$ROOT/data/smoke"}"    # directory of Harbor task dirs (each with instruction.md)
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
CHAT_TEMPLATE=/opt/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2

: "${APPTAINER_BRIDGE_URL:?not set - submit_2node.sbatch reads it from \$ROOT/bridge.url}"

NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
NUM_ENGINES=$((NUM_GPUS / TP))      # colocate_all requires num_engines * TP == policy GPUs
DP=$((NUM_GPUS / SP))               # data-parallel size after sequence parallel

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
#   8 tasks x 4 samples = 32 trajectories/step; mini 8 -> 8*4/8 = 4 per GCD at SP=2 (2 at SP=4),
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
  trainer.placement.colocate_all=true \
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
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE \
  generator.inference_engine.max_num_seqs=8 \
  generator.inference_engine.max_num_batched_tokens=8192 \
  generator.inference_engine.gpu_memory_utilization=0.60 \
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
  trainer.run_name="harbor_lumi_${NUM_NODES}node_${SERVED_NAME}_sp${SP}" \
  trainer.resume_mode=null \
  trainer.log_path="$ROOT/logs" \
  trainer.ckpt_path="$ROOT/ckpts/harbor_${SERVED_NAME}_ckpt" \
  "$@"