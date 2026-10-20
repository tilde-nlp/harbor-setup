set -x

# ============================ EDIT HERE ============================
# cluster
: "${NUM_NODES:=${SLURM_JOB_NUM_NODES:-4}}"
: "${GPUS_PER_NODE:=8}"
: "${POLICY_NODES:=2}"                       
: "${INFER_NODES:=2}"
: "${REF_NODES:=0}" # POLICY_NODES + INFER_NODES + REF_NODES == NUM_NODES

# parallelism
: "${TP:=4}"                                 # vLLM tensor parallel per engine; must divide GPUS_PER_NODE
: "${SP:=4}"                                 # Ulysses sequence parallel for the policy; must divide policy GPUs

# model
: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
: "${MODEL:=$ROOT/tildeopen}"                # local dir or hub id (HF_HUB_OFFLINE=1 is set, so pre-download)
: "${SERVED_NAME:=$(basename "$MODEL")}"     # no '/' allowed: Harbor addresses hosted_vllm/<name>
: "${CHAT_TEMPLATE:=/opt/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"

# context
: "${MAX_MODEL_LEN_CAP:=65536}"              # final MAX_MODEL_LEN = min(cap, max_position_embeddings from config.json)
: "${MAX_GEN_LEN:=2048}"                     # per LLM call (one agent turn), not per trajectory
: "${MAX_TURNS:=2}"

# rollout
: "${DATA_DIR:="$ROOT/data/"}"               # data directory
: "${N_CONCURRENT:=8}"                       # concurrent Harbor trials; <= workers in the env job
: "${GPU_MEM_UTIL:=0.80}"                    # vLLM KV cache share
: "${MAX_NUM_SEQS:=8}"                       # per vLLM engine
: "${MAX_BATCHED_TOKENS:=16384}"             # per vLLM engine
: "${INFERENCE_BACKEND:=vllm}"

# training
: "${TRAIN_BS:=4}"                           # distinct tasks per step; multiple of MINI_BS
: "${MINI_BS:=4}"                            # tasks per optimizer step
: "${N_SAMPLES:=4}"                          # trajectories per task
: "${MICRO_TRAIN_BS:=1}"                     # per DP rank, with grads
: "${MICRO_FWD_BS:=1}"                       # per DP rank, forward only

# logging
: "${LOGGER:=console}"
# ===================================================================

# --- derived ---
die() { echo "FATAL: $*" >&2; exit 1; }

POLICY_GPUS=$(( POLICY_NODES * GPUS_PER_NODE ))
REF_GPUS=$(( REF_NODES * GPUS_PER_NODE ))
INFER_GPUS=$(( INFER_NODES * GPUS_PER_NODE ))
NUM_ENGINES=$(( INFER_GPUS / TP ))           # vLLM replicas
DP=$(( POLICY_GPUS / SP ))                   # FSDP data parallel ranks

MODEL_CTX=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["max_position_embeddings"])' "$MODEL" 2>/dev/null || echo 32768)
MAX_MODEL_LEN=$(( MODEL_CTX < MAX_MODEL_LEN_CAP ? MODEL_CTX : MAX_MODEL_LEN_CAP ))

# layout
(( POLICY_NODES >= 1 && INFER_NODES >= 1 ))     || die "need at least one policy node and one inference node"
(( POLICY_NODES + INFER_NODES + REF_NODES == NUM_NODES ))   || die "POLICY_NODES ($POLICY_NODES) + INFER_NODES ($INFER_NODES) != NUM_NODES ($NUM_NODES)"
(( GPUS_PER_NODE % TP == 0 ))                   || die "TP=$TP must divide GPUS_PER_NODE=$GPUS_PER_NODE (an engine never spans nodes)"
(( POLICY_GPUS % SP == 0 ))                     || die "SP=$SP must divide policy GPUs=$POLICY_GPUS"
(( REF_GPUS % SP == 0 ))                        || die "SP=$SP must divide ref GPUs=$REF_GPUS"
(( DP >= 2 ))                                   || die "DP=$DP: lower SP or raise POLICY_NODES"

# batch rule (skyrl/train/utils/utils.py): MINI_BS * N_SAMPLES / DP must be an integer divisible by MICRO_TRAIN_BS
(( TRAIN_BS % MINI_BS == 0 ))                   || die "TRAIN_BS=$TRAIN_BS must be a multiple of MINI_BS=$MINI_BS"
(( MINI_BS * N_SAMPLES % DP == 0 ))             || die "MINI_BS*N_SAMPLES=$(( MINI_BS * N_SAMPLES )) not divisible by DP=$DP"
(( MINI_BS * N_SAMPLES / DP % MICRO_TRAIN_BS == 0 )) || die "per-rank batch $(( MINI_BS * N_SAMPLES / DP )) not divisible by MICRO_TRAIN_BS=$MICRO_TRAIN_BS"

# context
(( MAX_GEN_LEN < MAX_MODEL_LEN ))               || die "MAX_GEN_LEN=$MAX_GEN_LEN must be below MAX_MODEL_LEN=$MAX_MODEL_LEN"

echo "layout: ${NUM_NODES}n | policy ${POLICY_NODES}n/${POLICY_GPUS} GCDs SP=$SP DP=$DP | vLLM ${INFER_NODES}n/${INFER_GPUS} GCDs TP=$TP x $NUM_ENGINES engines | ctx $MAX_MODEL_LEN | $TRAIN_BS tasks x $N_SAMPLES samples, $(( MINI_BS * N_SAMPLES / DP )) per rank"

# for up to date information use the SkyRL/skyrl/train/config/config.py file, instead of docs
ARGS=(
  # --- DATA ---

  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \

  # --- TRAINER PLACEMENT ---

  trainer.algorithm.advantage_estimator="grpo"
  # gae, grpo, rloo, reinforce_pp, maxrl, or custom via AdvantageEstimatorRegistry

  trainer.policy.model.path="$MODEL"
  # path to model (hf style), or HuggingFace name (with HF_HOME defined)

  trainer.placement.colocate_all=false
  # an option to share the gpus for inferenence/policy/critics etc. (WE DON'T USE colocate_all=true)

  trainer.strategy=fsdp
  # distributed (multi-gpu/multi-node) training backend (another option - megatron, but that needs additional dependencies for this container)

  trainer.policy.sequence_parallel_size=$SP
  trainer.ref.sequence_parallel_size=$SP

  trainer.policy.fsdp_config.cpu_offload=false
  # offloads some params to RAM (avoid it like the plague)

  trainer.policy.fsdp_config.reshard_after_forward=true
  trainer.gradient_checkpointing=true
  trainer.logprobs_chunk_size=512

  trainer.placement.policy_num_nodes=$POLICY_NODES
  trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE

  trainer.placement.ref_num_nodes=$REF_NODES
  trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE
  trainer.placement.colocate_policy_ref=false
  # reference model is not used with grpo or with kl_loss=false
  # does nothing in current setup
  # BTW: there's an option to colocate ref with policy

  trainer.placement.critic_num_nodes=$POLICY_NODES
  trainer.placement.critic_num_gpus_per_node=$GPUS_PER_NODE
  # TODO: haven't checked how critic works yet
  # isn't used without trainer.critic.model.path
  # does nothing in current setup

  # --- INFERENCE ---

  generator.inference_engine.served_model_name="$SERVED_NAME"

  generator.inference_engine.num_engines=$NUM_ENGINES
  # how many inference engines to start (num_nodes * num_gpus_per_node = num_engines * tp * dp * pp)

  generator.inference_engine.tensor_parallel_size=$TP
  # tensor parallel for each engine (split the inference engine (model instance) between how many gpus)

  generator.inference_engine.distributed_executor_backend=mp
  # most examples in SkyRL use mp (python multiprocessing), the gsm8k setup is tested with mp. USING ray MAY CAUSE ERRORS

  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN
  # prompt + model's output <= max_model_len

  generator.inference_engine.gpu_memory_utilization=$GPU_MEM_UTIL
  # use less for colocate_all=true

  generator.inference_engine.backend=$INFERENCE_BACKEND

  generator.inference_engine.run_engines_locally=true
  # SkyRL launches inference engines; otherwise - can pass url of already launched with generator.inference_engine.external_server_urls

  generator.inference_engine.weight_sync_backend=nccl
  # the strategy of copying new weights (after optimizer step) to vLLM

  generator.batched=false
  # use false for agentic tasks, can use true for single-step tasks. TODO: understand

  generator.n_samples_per_prompt=$N_SAMPLES
  # copy initial prompts that many times - create different trajectories due to the temperature. advantage is calculated by using all trajectories for this task (basically, model learns to choose the best response from the ones it generated)

  generator.sampling_params.max_generate_length=$MAX_GEN_LEN
  # maximum output length, if no END token generated - vllm will cut off the generation of this sequence
  # cap per LLM call (one agent turn), not per trajectory

  # --- HYPER PARAMETERS ---

  trainer.epochs=1

  trainer.train_batch_size=$TRAIN_BS
  # how many DISTINCT tasks for each inference steps
  # keep in mind that it will generate train_batch_size*n_samples_per_prompt trajectories

  trainer.policy_mini_batch_size=$MINI_BS
  # how many prompts (NOT TRAJECTORIES) will be used for each optimizer.step()
  # by prompt I mean n_samples_per_prompt trajectories for a single task
  # when train_batch_size/policy_mini_batch_size = 2, that means for each train_batch there are 2 optimizer.step()
  # policy_mini_batch_size <= train_batch_size

  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BS
  # grad accumulation of policy_mini_batch_size per data parallel rank (not gpu)
  # there will be (policy_mini_batch_size * n_samples_per_prompt / DP) / micro_train_batch_size_per_gpu passes of accumulating gradients until optimizer.step()

  trainer.micro_forward_batch_size_per_gpu=$MICRO_FWD_BS
  # grad accumulation of FORWARD ONLY step (policy old_logprobs, critic, ref, everything else that doesn't need gradients)
  # policy_mini_batch_size/micro_forward_batch_size_per_gpu steps, but since no grads are needed, this can be bigger than micro_train_batch_size_per_gpu
  # first we run forward only, then run forward+backward (account with grad_acc) then happens optimizer.step()

  trainer.update_epochs_per_batch=1
  # how many times to go over the same train batch (trainer.train_batch_size) before getting another batch

  trainer.max_tokens_per_microbatch=65536
  # IMPORTANT 

  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN
  trainer.eval_batch_size=$TRAIN_BS
  trainer.eval_before_train=false
  trainer.eval_interval=-1
  trainer.ckpt_interval=-1
  trainer.policy.optimizer_config.lr=1.0e-6
  trainer.algorithm.use_kl_loss=false

  # --- UTILITIES ---

  trainer.logger="$LOGGER"
  trainer.project_name="gsm8k"
  trainer.run_name="gsm8k_lumi_${NUM_NODES}n_${SERVED_NAME}_sp${SP}_dp${DP}"
  trainer.resume_mode=null
  trainer.log_path="$ROOT/logs"
  trainer.ckpt_path="$ROOT/ckpts/gsm8k_${SERVED_NAME}_ckpt"
)

python -m skyrl.train.entrypoints.main_base "${ARGS[@]}" "$@"