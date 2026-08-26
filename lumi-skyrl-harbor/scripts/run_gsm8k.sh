set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/train/gsm8k/run_gsm8k.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

# You can override the default values with e.g.: `NUM_GPUS=1 bash examples/train/gsm8k/run_gsm8k.sh`.

export ROOT="/scratch/project_465003062/environment/containers/skyrl-harbor"

: "${DATA_DIR:="$ROOT/data/gsm8k"}"
: "${TRAIN_GPUS:=2}"
: "${INFER_GPUS:=2}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"

uv run --active --no-sync --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.policy.megatron_config.tensor_model_parallel_size=2 \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAIN_GPUS \
  generator.inference_engine.num_engines=$INFER_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.max_model_len=2048 \
  generator.inference_engine.distributed_executor_backend=mp \
  trainer.epochs=20 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.offload_after_step=false \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-smoke" \
  trainer.run_name="gsm8k-smoke_test" \
  trainer.resume_mode=null \
  trainer.log_path="$ROOT/skyrl-logs" \
  trainer.ckpt_path="$ROOT/ckpts/gsm8k-smoke_4B_ckpt" \
  $@
