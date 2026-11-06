#!/bin/bash
#SBATCH --job-name=skyrl-replay
#SBATCH --account=project_465003062
#SBATCH --partition=standard-g
#SBATCH --nodes=4                      # POLICY_NODES + REF_NODES (no inference node: engines are never started)
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00                # TRAIN_BS=5: ~1,300 steps; the per-step floor (param all-gathers, grad reduce) dominates
#SBATCH --output=%x-%j.out
#
# Offline GRPO on stored rollouts (replay_lean/) with SkyRL + FSDP on LUMI.
# Roles: sbatch (first node) -> srun one `node` task per node -> container -> `inner` (ray symmetric-run)
#        -> `train` on the head node. Every VAR in config() can be overridden by exporting it before sbatch:
#   REF_NODES=2 sbatch --nodes=6 skyrl_replay.sbatch       # reference on its own nodes instead of colocated
#   EXTRA_ARGS="trainer.dump_data_batch=true" sbatch ...     # extra SkyRL overrides
#   EXTRA_ENV="NCCL_NET=Socket NCCL_DEBUG=INFO" sbatch ...   # extra environment (no spaces in values)

config() {
  set -a
  # layout: 4 policy nodes = 32 GCDs, SP=4 -> DP=8. REF_NODES=0 colocates the frozen reference with the
  # policy (about 2 GB per GCD sharded). No inference node: main_replay.py never starts the engines.
  : "${POLICY_NODES:=4}"; : "${REF_NODES:=0}"; : "${GPUS_PER_NODE:=8}"; : "${SP:=4}"
  # paths
  : "${ROOT:=/scratch/project_465003062/environment/containers/grpo-pregen-rl}"
  : "${SANDBOX:=$ROOT/multitorch_sandbox.sif}"
  : "${BINDS:=/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038}"
  : "${MODEL_SRC:=$ROOT/tildeopen}"                      # local HF dir (HF_HUB_OFFLINE=1)
  : "${REPLAY_PARQUET:=$ROOT/data/replay/replay.parquet}" # python -m replay.prepare_replay_data ...
  # batch: 6,488 prompts x 8 rollouts, 5 prompts (40 rollouts, ~52k loss tokens) per optimizer step.
  # 6,488 / 5 = 1,297 steps; MAX_STEPS=1296 stops on the last HF export (3 x 432).
  : "${TRAIN_BS:=5}"; : "${N_SAMPLES:=8}"; : "${MAX_SEQ_LEN:=65536}"; : "${MAX_STEPS:=1296}"
  : "${PROJECT_NAME:=replay_bs${TRAIN_BS}}"   # ckpt and export dirs are keyed on this; one dir per batch-size setting
  : "${LOG_DIR:=$ROOT/logs}"; : "${CKPT_DIR:=$ROOT/ckpts}"; : "${RUNS_DIR:=$ROOT/previous_runs}"
  : "${EXPORT_PATH:=$ROOT/exports/$PROJECT_NAME}"
  # loss: rollout_is (ratio against the reference worker, band 1 +- EPS_CLIP) + k3 KL to the same reference
  : "${EPS_CLIP:=0.2}"; : "${KL_COEF:=0.05}"
  # optimizer: identical to the TRAIN_BS=40 baseline so the batch size is the only difference.
  # If the band masks most tokens late in the pass, the retuned arm is: LR=3e-7 WARMUP_STEPS=60 ADAM_BETA2=0.9999
  : "${LR:=1.0e-6}"; : "${WARMUP_STEPS:=10}"; : "${ADAM_BETA2:=0.999}"
  : "${HF_SAVE_INTERVAL:=432}"           # bf16 HF export for eval at 1/3, 2/3, 3/3 of the data (54/108/162 in the baseline)
  : "${CKPT_INTERVAL:=108}"              # FSDP checkpoint for resume (~480 GB, one kept); ~1 h of work at risk on a time-out
  : "${RESUME_MODE:=null}"               # always start fresh; after a time-out: RESUME_MODE=latest sbatch skyrl_replay.sbatch
  : "${LOGGER:=console}"
  : "${EXTRA_ARGS:=}"; : "${EXTRA_ENV:=}"; : "${RAY_PORT:=6379}"
  set +a
}

die() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------- train (driver, head node, in container)
role_train() {
  POLICY_GPUS=$(( POLICY_NODES * GPUS_PER_NODE )); DP=$(( POLICY_GPUS / SP ))
  (( POLICY_NODES + REF_NODES == SLURM_JOB_NUM_NODES )) || die "POLICY_NODES+REF_NODES != $SLURM_JOB_NUM_NODES nodes"
  (( POLICY_GPUS % SP == 0 ))                            || die "SP=$SP does not divide $POLICY_GPUS policy GCDs"
  (( TRAIN_BS * N_SAMPLES % DP == 0 ))                   || die "TRAIN_BS*N_SAMPLES=$(( TRAIN_BS * N_SAMPLES )) not divisible by DP=$DP"
  if (( REF_NODES == 0 )); then COLOCATE_REF=true; REF_PLACE_NODES=$POLICY_NODES
  else COLOCATE_REF=false; REF_PLACE_NODES=$REF_NODES; fi
  echo "layout: policy ${POLICY_NODES}n SP=$SP DP=$DP | ref ${REF_PLACE_NODES}n colocated=$COLOCATE_REF | no inference engines"

  # shellcheck disable=SC2086
  python -m replay_lean.main_replay \
    data.train_data="['$REPLAY_PARQUET']" \
    trainer.policy.model.path="$MODEL" \
    trainer.strategy=fsdp \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=$POLICY_NODES trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE \
    trainer.placement.ref_num_nodes=$REF_PLACE_NODES   trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE \
    trainer.placement.colocate_policy_ref=$COLOCATE_REF \
    trainer.policy.optimizer_config.offload_after_step=false \
    trainer.policy.fsdp_config.fsdp_size=$POLICY_GPUS \
    trainer.policy.sequence_parallel_size=$SP trainer.ref.sequence_parallel_size=$SP \
    trainer.gradient_checkpointing=true trainer.logprobs_chunk_size=512 \
    trainer.remove_microbatch_padding=true trainer.max_tokens_per_microbatch=$MAX_SEQ_LEN \
    trainer.algorithm.max_seq_len=$MAX_SEQ_LEN \
    generator.n_samples_per_prompt=$N_SAMPLES generator.sampling_params.temperature=1.0 \
    trainer.epochs=1 trainer.max_training_steps=$MAX_STEPS \
    trainer.train_batch_size=$TRAIN_BS trainer.policy_mini_batch_size=$TRAIN_BS \
    trainer.micro_train_batch_size_per_gpu=1 trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.algorithm.advantage_estimator=grpo trainer.algorithm.grpo_norm_by_std=false \
    trainer.algorithm.loss_reduction=token_mean \
    trainer.algorithm.policy_loss_type=rollout_is \
    trainer.algorithm.eps_clip_low=$EPS_CLIP trainer.algorithm.eps_clip_high=$EPS_CLIP \
    trainer.algorithm.use_kl_loss=true trainer.algorithm.kl_loss_coef=$KL_COEF trainer.algorithm.kl_estimator_type=k3 \
    trainer.policy.optimizer_config.lr=$LR trainer.policy.optimizer_config.num_warmup_steps=$WARMUP_STEPS \
    trainer.policy.optimizer_config.adam_betas="[0.9,$ADAM_BETA2]" \
    trainer.eval_before_train=false trainer.eval_interval=-1 \
    trainer.ckpt_interval=$CKPT_INTERVAL trainer.max_ckpts_to_keep=1 trainer.hf_save_interval=$HF_SAVE_INTERVAL \
    trainer.resume_mode=$RESUME_MODE trainer.ckpt_path="$CKPT_DIR/${PROJECT_NAME}_$(basename "$MODEL_SRC")" \
    trainer.export_path="$EXPORT_PATH" trainer.log_path="$LOG_DIR" \
    trainer.logger="$LOGGER" trainer.project_name="$PROJECT_NAME" trainer.run_name="${PROJECT_NAME}_${SLURM_JOB_ID}" \
    $EXTRA_ARGS "$@"
}

# ---------------------------------------------------------------- environment (every node, in container)
apply_env() {
  . /opt/venv/bin/activate
  export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:$ROOT/pyhooks${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HOME="$ROOT/hf" HF_HUB_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=true HF_PARALLEL_LOADING_WORKERS=8
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"; unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export HWLOC_COMPONENTS=-amd_smi,-rsmi
  export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-$SLURM_JOB_ID" TORCH_EXTENSIONS_DIR="/tmp/$USER-torch-ext-$SLURM_JOB_ID"
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH" "$TORCH_EXTENSIONS_DIR"
  export RAY_USAGE_STATS_ENABLED=0 RAY_DEDUP_LOGS=0 RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=300 RAY_py_gcs_connect_timeout_s=300
  export SKYRL_LD_LIBRARY_PATH_EXPORT=True SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600   # rank 0 loads the checkpoint alone, ~10 min
  export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=7
  # Slingshot via libfabric (aws-ofi-nccl). Every line was needed on LUMI; see the previous script for the history.
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 GLOO_SOCKET_IFNAME=hsn0 CXI_FORK_SAFE=1 CXI_FORK_SAFE_HP=1
  export RCCL_MSCCL_ENABLE=0 RCCL_MODEL_MATCHING_DISABLE=1       # was for the weight-sync communicator; harmless to keep
  export NCCL_NET=Libfabric NCCL_NET_GDR_LEVEL=PHB
  if [ -f /usr/lib/x86_64-linux-gnu/librccl-net.so ]; then export NCCL_NET_PLUGIN=/usr/lib/x86_64-linux-gnu/librccl-net.so; fi
  export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DISABLE_CQ_HUGETLB=1
  export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1                      # required: all-reduces hang over Libfabric without it
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_DEBUG_FILE="$LOG_DIR/nccl-$SLURM_JOB_ID/%h.%p.log"
  mkdir -p "$LOG_DIR/nccl-$SLURM_JOB_ID"
  if [ -n "$EXTRA_ENV" ]; then export $EXTRA_ENV; echo "[$(hostname)] EXTRA_ENV: $EXTRA_ENV"; fi
}

# ---------------------------------------------------------------- launch plumbing
role_batch() {
  # The other nodes and the container run the archived copy on /scratch, not Slurm's private copy.
  RUN_DIR="$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"; mkdir -p "$RUN_DIR"
  export SELF="$RUN_DIR/skyrl_replay.sbatch"; cp "${BASH_SOURCE[0]}" "$SELF"
  scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/slurm_job.txt"
  env | grep -E '^(POLICY_NODES|REF_NODES|SP|TRAIN_BS|N_SAMPLES|MAX_SEQ_LEN|MAX_STEPS|EPS_CLIP|KL_COEF|LR|WARMUP_STEPS|ADAM_BETA2|HF_SAVE_INTERVAL|CKPT_INTERVAL|RESUME_MODE|PROJECT_NAME|MODEL_SRC|REPLAY_PARQUET|EXTRA_ARGS|EXTRA_ENV)=' > "$RUN_DIR/config.txt"
  LOG_FILE="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  trap 'rc=$?; cp -f "$LOG_FILE" "$RUN_DIR/" 2>/dev/null; exit $rc' EXIT
  trap 'exit 143' TERM INT HUP

  HEAD_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
  export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
  echo "run archive: $RUN_DIR   head: $(hostname) ($RAY_ADDRESS)"
  srun --label --kill-on-bad-exit=1 --mem=0 --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
       --ntasks-per-node=1 --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$SELF" node
}

role_node() {
  # One task per node, on the host. Check GPUs, stage the checkpoint to node-local RAM, enter the container.
  ngpu=$(tr ',' '\n' <<< "${ROCR_VISIBLE_DEVICES:-}" | grep -c .)
  (( ngpu == GPUS_PER_NODE )) || die "[$SLURMD_NODENAME] got $ngpu/$GPUS_PER_NODE GPUs"
  export MODEL="/tmp/$USER-model/$(basename "$MODEL_SRC")"; mkdir -p "$MODEL"
  ls "$MODEL_SRC" | xargs -P 16 -I{} cp -rL "$MODEL_SRC/{}" "$MODEL/"
  exec singularity exec -B "$BINDS" "$SANDBOX" bash "$SELF" inner
}

role_inner() {
  apply_env
  ray stop --force >/dev/null 2>&1 || true
  exec ray symmetric-run --address "$RAY_ADDRESS" --min-nodes "$SLURM_JOB_NUM_NODES" \
       --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" -- bash "$SELF" train
}

# ---------------------------------------------------------------- dispatch
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then config; apply_env; return 0; fi   # sourced for debugging
set -euo pipefail
config
case "${1:-batch}" in
  batch) role_batch ;;
  node)  role_node ;;
  inner) role_inner ;;
  train) shift; role_train "$@" ;;
  *)     die "usage: sbatch $0  (roles: node | inner | train)" ;;
esac
