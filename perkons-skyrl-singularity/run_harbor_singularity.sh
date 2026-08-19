#!/usr/bin/env bash
# run_harbor_singularity.sh
set -euo pipefail

# export WANDB_API_KEY=YOUR_KEY_HERE

#-----------------------
# Container setup
#-----------------------
WORK=/local_data/rusevich/llm/skyrl-singularity
SIF="$WORK/sif/skyrl.sif"
REPO_DIR="$WORK/SkyRL"
GPUS=3,4,5

#-----------------------
# Dataset setup
#-----------------------
DATA_DIR="$WORK/harbor_data"
TRAIN_DATA="['$DATA_DIR/CodeContests']"
EVAL_DATA="['$DATA_DIR/eval-small']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="codecontest"
STORAGE_ROOT="$WORK/runs/$RUN_NAME"
TRIALS_DIR="$STORAGE_ROOT/trials"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

# Config group: $HYDRA_SEARCHPATH/harbor_trial_config/$HARBOR_TRIAL_CONFIG.yaml
HARBOR_TRIAL_CONFIG="singularity"

#-----------------------
# Training setup
#-----------------------
MODEL_PATH="Qwen/Qwen3-1.7B"
SERVED_NAME="Qwen3-1.7B"

N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=4
MAX_MODEL_LEN=32768
EVAL_INTERVAL=5
LR=1.0e-6

ADVANTAGE_ESTIMATOR="grpo"
APPLY_OVERLONG_FILTERING=true

#-----------------------
# Infrastructure setup
#-----------------------
NUM_POLICY_GPUS=2
NUM_INFERENCE_ENGINES=1
TP_SIZE=1
COLOCATE_ALL=false

ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=1
MAX_CONCURRENCY=8

NUM_REF_GPUS=1

#### DEBUG
UV_HTTP_TIMEOUT=15

mkdir -p "$WORK/home" "$TRIALS_DIR" "$CKPTS_DIR" "$EXPORTS_DIR" "$LOG_DIR" "$WORK/ray_tmp"

singularity exec \
  --nv \
  --home "$WORK/home:$WORK/home" \
  --bind "$WORK:$WORK" \
  --bind /etc/resolv.conf:/etc/resolv.conf \
  --bind "$WORK/ray_tmp:/ray" \
  --env RAY_TMPDIR=/ray \
  --env CUDA_VISIBLE_DEVICES="$GPUS" \
  --env HOME="$WORK/home" \
  --env HF_HOME="$WORK/hf_cache" \
  --env UV_CACHE_DIR="$WORK/uv_cache" \
  --env APPTAINER_USERNS=1 \
  --env APPTAINER_TMPDIR="$WORK/apptainer_tmp" \
  --env APPTAINER_CACHEDIR="$WORK/apptainer_tmp/cache" \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  "$SIF" bash -s -- "$@" <<EOF
set -euo pipefail
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

ray stop --force >/dev/null 2>&1 || true
ray start --head --temp-dir=/ray --disable-usage-stats
cd $REPO_DIR

uv run --isolated --extra fsdp --extra harbor \
  -m examples.train_integrations.harbor.entrypoints.main_harbor \
  harbor_trial_config=$HARBOR_TRIAL_CONFIG \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.ref.fsdp_config.reshard_after_forward=true \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.placement.colocate_all=$COLOCATE_ALL \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_REF_GPUS \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.eval_before_train=false \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  generator.step_wise_trajectories=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.served_model_name=$SERVED_NAME \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  \$@
EOF