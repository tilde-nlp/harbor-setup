set -x

# Colocated GRPO training+generation for Qwen3-14B on GSM8K, multi-node LUMI / ROCm.
# Launched by submit_2node.sbatch on the Ray head node, inside the sandbox, after skyrl_env.sh.
#
# 2 nodes x 8 GCDs = 16 GPUs: FSDP shards policy/ref over all 16, vLLM runs 8 engines with TP=2
# (14B bf16 = 29.6 GB -> 14.8 GB weights per GCD, ~25 GB KV cache at 0.65 utilization).
#
# For Qwen3-32B use 4 nodes and TP=4 (nodes=4, MODEL=Qwen/Qwen3-32B, TP=4): 65.6 GB bf16 -> 16.4 GB
# weights per GCD, and the CPU-offloaded optimizer state (~525 GB total) spreads over 4 x 512 GB nodes.
# 32B on 2 nodes fits on the GCDs but is tight on host RAM.

: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
: "${DATA_DIR:="$ROOT/data"}"
: "${NUM_NODES:=${SLURM_JOB_NUM_NODES:-2}}"
: "${GPUS_PER_NODE:=8}"
: "${MODEL:=Qwen/Qwen3-14B}"        # pre-download into $HF_HOME: HF_HUB_OFFLINE=1 is set
: "${TP:=2}"                        # vLLM tensor parallel per engine; must divide GPUS_PER_NODE
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"

NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
NUM_ENGINES=$((NUM_GPUS / TP))      # colocate_all requires num_engines * TP == policy GPUs

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

# Batch-size rules with dp=16: policy_mini_batch_size * n_samples_per_prompt / NUM_GPUS must be a
# positive integer divisible by micro_train_batch_size_per_gpu (32*4/16 = 8, micro 2 -> ok).
python -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.critic_num_nodes=$NUM_NODES \
  trainer.placement.critic_num_gpus_per_node=$GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.engine_init_kwargs.max_model_len=4096 \
  generator.inference_engine.gpu_memory_utilization=0.65 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.max_generate_length=1024 \
  environment.env_class=gsm8k \
  trainer.epochs=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=512 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_lumi_${NUM_NODES}node_$(basename "$MODEL")" \
  trainer.resume_mode=null \
  trainer.log_path="$ROOT/logs" \
  trainer.ckpt_path="$ROOT/ckpts/gsm8k_$(basename "$MODEL")_ckpt" \
  "$@"
