set -x

# ============================ EDIT HERE ============================
: "${COLOCATE:=true}"                       # true: policy + vLLM share all GPUs; false: separate nodes
: "${NUM_NODES:=${SLURM_JOB_NUM_NODES:-2}}"
: "${GPUS_PER_NODE:=8}"
: "${POLICY_NODES:=1}"                       # ignored when COLOCATE=true (policy takes all nodes)
: "${INFER_NODES:=1}"                        # ignored when COLOCATE=true (vLLM takes all nodes)
: "${TP:=2}"                                 # vLLM tensor parallel per engine; must divide GPUS_PER_NODE
: "${GPU_MEM_UTIL:=0.6}"                     # use ~0.5-0.65 when COLOCATE=true
: "${MAX_MODEL_LEN:=4096}"                   # prompt + generation, must be >= max_prompt_length + max_generate_length
: "${MODEL:=Qwen/Qwen3-14B}"                 # pre-download into $HF_HOME: HF_HUB_OFFLINE=1 is set
: "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
: "${DATA_DIR:="$ROOT/data"}"
: "${LOGGER:=console}"
# ===================================================================

# --- derived, do not edit ---
if [ "$COLOCATE" = true ]; then
  POLICY_NODES=$NUM_NODES
  INFER_NODES=$NUM_NODES
elif (( POLICY_NODES + INFER_NODES != NUM_NODES )); then
  echo "POLICY_NODES ($POLICY_NODES) + INFER_NODES ($INFER_NODES) != NUM_NODES ($NUM_NODES)" >&2
  exit 1
fi
INFER_GPUS=$((INFER_NODES * GPUS_PER_NODE))
if (( INFER_GPUS % TP != 0 )); then
  echo "TP=$TP does not divide inference GPUs=$INFER_GPUS" >&2
  exit 1
fi
NUM_ENGINES=$((INFER_GPUS / TP))             # num_engines * TP == inference GPUs (== policy GPUs when colocated)

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

# for up to date information use the SkyRL/skyrl/train/config/config.py file, instead of docs

ARGS=(
  # --- DATA ---

  data.train_data="['$DATA_DIR/train.parquet']"                        
  data.val_data="['$DATA_DIR/validation.parquet']"                     
  
  # --- TRAINER PLACEMENT ---

  trainer.algorithm.advantage_estimator="grpo"     
  # gae, grpo, rloo, reinforce_pp, maxrl, or custom via AdvantageEstimatorRegistry

  trainer.policy.model.path="$MODEL"
  # path to model (hf style), or HuggingFace name (with HF_HOME defined)

  trainer.placement.colocate_all=$COLOCATE
  # an option to share the gpus for inferenence/policy/critics etc. (we don't use this, but it's easy for debugging)

  trainer.strategy=fsdp
  # distributed (multi-gpu/multi-node) training backend (another option - megatron, but that needs additional dependencies for this container)

  trainer.policy.fsdp_config.cpu_offload=false
  # offloads some params to RAM (avoid it like the plague)

  trainer.placement.policy_num_nodes=$POLICY_NODES
  trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE

  trainer.placement.ref_num_nodes=$POLICY_NODES                           
  trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE  
  # reference model is not used with grpo or with kl_loss=false
  # does nothing in current setup
  # BTW: there's an option to colocate ref with policy

  trainer.placement.critic_num_nodes=$POLICY_NODES                        
  trainer.placement.critic_num_gpus_per_node=$GPUS_PER_NODE   
  # TODO: haven't checked how critic works yet
  # isn't used without trainer.critic.model.path
  # does nothing in current setup
  
  # --- INFERENCE ---
  
  generator.inference_engine.num_engines=$NUM_ENGINES
  # how many inference engines to start (num_nodes * num_gpus_per_node = num_engines * tp * dp * pp)

  generator.inference_engine.tensor_parallel_size=$TP
  # tensor parallel for each engine (split the inference engine (model instance) between how many gpus)

  generator.inference_engine.distributed_executor_backend=mp
  # most examples in SkyRL use mp (python multiprocessing), this setup is tested with mp. other option - ray
  
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN
  # prompt + model's output <= max_model_len

  generator.inference_engine.gpu_memory_utilization=$GPU_MEM_UTIL
  # use less for colocate_all=true

  generator.inference_engine.backend="vllm"

  generator.inference_engine.run_engines_locally=true
  # SkyRL launches inference engines; otherwise - can pass url of already launched with generator.inference_engine.external_server_urls
  
  generator.inference_engine.weight_sync_backend=nccl
  # the strategy of copying new weights (after optimizer step) to vLLM  

  generator.batched=false
  # use false for agentic tasks, can use true for single-step tasks. TODO: understand

  generator.chat_template_kwargs.enable_thinking=false
  # kwargs passed to tokenizer.apply_chat_template, this one is qwen specific

  generator.n_samples_per_prompt=4
  # copy initial prompts that many times - create different trajectories due to the temperature. advantage is calculated by using all trajectories for this task (basically, model learns to choose the best response from the ones it generated)

  generator.sampling_params.max_generate_length=1024
  # maximum output length, if no END token generated - vllm will cut off the generation of this sequence
  
  # --- HYPER PARAMETERS ---
  
  environment.env_class=gsm8k
  trainer.epochs=1
                                                 
  trainer.train_batch_size=64
  # how many DISTINCT tasks for each inference steps
  # keep in mind that it will generate train_batch_size*n_samples_per_prompt trajectories

  trainer.policy_mini_batch_size=32   
  # how many prompts (NOT TRAJECTORIES) will be used for each optimizer.step()
  # by prompt I mean n_samples_per_prompt trajectories for a single task
  # when train_batch_size/policy_mini_batch_size = 2, that means for each train_batch there are 2 optimizer.step()
  # policy_mini_batch_size <= train_batch_size   

  trainer.micro_train_batch_size_per_gpu=2
  # grad accumulation of policy_mini_batch_size per data parallel rank (not gpu)
  # there will be policy_mini_batch_size/micro_train_batch_size_per_gpu passes of accumulating gradients until optimizer.step()

  trainer.micro_forward_batch_size_per_gpu=4
  # grad accumulation of FORWARD ONLY step (policy old_logprobs, critic, ref, everything else that doesn't need gradients)
  # policy_mini_batch_size/micro_forward_batch_size_per_gpu steps, but since no grads are needed, this can be bigger than micro_train_batch_size_per_gpu

  trainer.update_epochs_per_batch=1
  # how many times to go over the same train batch (trainer.train_batch_size) before getting another batch

  trainer.max_prompt_length=512
  trainer.eval_batch_size=32
  trainer.eval_before_train=false
  trainer.eval_interval=-1
  trainer.ckpt_interval=-1
  trainer.policy.optimizer_config.lr=1.0e-6
  trainer.algorithm.use_kl_loss=false
  
  # --- UTILITIES ---
  
  trainer.logger="$LOGGER"
  trainer.project_name="gsm8k"
  trainer.run_name="gsm8k_lumi_${NUM_NODES}node_$(basename "$MODEL")"  
  trainer.resume_mode=null
  trainer.log_path="$ROOT/logs"
  trainer.ckpt_path="$ROOT/ckpts/gsm8k_$(basename "$MODEL")_ckpt"
)

python -m skyrl.train.entrypoints.main_base "${ARGS[@]}" "$@"