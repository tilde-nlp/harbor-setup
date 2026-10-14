cat > /opt/SkyRL/examples/train/gsm8k/run_4_gpus.sh << 'EOF'
set -x

# Colocated GRPO training+generation for Qwen3-4B-Instruct-2507 on GSM8K.
# LUMI / ROCm / 4 GCD variant of examples/train/gsm8k/run_gsm8k.sh.
#
# Run inside the sandbox, after:
#   salloc --account=project_465003062 --partition=dev-g \
#     --nodes=1 --gpus-per-node=4 --ntasks=1 --cpus-per-task=28 --mem=224G --time=2:00:00
#   srun --overlap --pty singularity run -B /var/spool/slurmd,$ROOT $ROOT/multitorch_sandbox bash
#   bash /opt/SkyRL/examples/train/gsm8k/run_gsm8k_lumi.sh

: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
: "${DATA_DIR:="$ROOT/data"}"
: "${NUM_GPUS:=8}"
: "${TRAIN_GPUS:=4}"
: "${ENGINE_GPUS:=4}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL:=Qwen/Qwen3-4B-Instruct-2507}"

# --- container env ---
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME="$ROOT/hf"
export HF_HUB_OFFLINE=1

# --- LUMI / ROCm env ---
# Ray workers do not inherit LD_LIBRARY_PATH, which carries ROCm + libfabric paths.
export SKYRL_LD_LIBRARY_PATH_EXPORT=True
# Slurm hands GPUs over via ROCR_VISIBLE_DEVICES; torch/vLLM read HIP_VISIBLE_DEVICES.
# export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT/pyhooks${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HWLOC_COMPONENTS=-amd_smi,-rsmi
# export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 - this makes ray allocate memory on one GPU and freak out
# export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1  
# Node-local scratch, reclaimed at job end.
export MIOPEN_USER_DB_PATH="/tmp/miopen-$USER"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-$USER"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0
# Single node, multi-GCD: keep RCCL off the network interfaces.
# export NCCL_SOCKET_IFNAME=lo #- this doesn't work
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_NET_PLUGIN=none
export VLLM_LOGGING_LEVEL=DEBUG

export NCCL_P2P_DISABLE=0

ray stop --force >/dev/null 2>&1
trap 'ray stop --force' EXIT
ray start --head --num-gpus="$NUM_GPUS" --num-cpus="${SLURM_CPUS_PER_TASK:-28}" --dashboard-host=0.0.0.0 --dashboard-port=8265


python -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS  \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS  \
  generator.inference_engine.num_engines=$NUM_GPUS  \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.max_model_len=4096 \
  generator.inference_engine.distributed_executor_backend=mp \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.65 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_lumi_4gcd" \
  trainer.resume_mode=null \
  trainer.log_path="$ROOT/logs" \
  trainer.ckpt_path="$ROOT/ckpts/gsm8k_4b_ckpt" \
  $@
EOF