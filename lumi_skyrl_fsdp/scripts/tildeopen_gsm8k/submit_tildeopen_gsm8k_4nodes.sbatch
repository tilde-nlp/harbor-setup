#!/bin/bash
#SBATCH --job-name=skyrl-tildeopen-gsm8k
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1           # one Ray process tree per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0                       # important on dev-g, OOM otherwise
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# SkyRL + FSDP on LUMI. One file, four roles, selected by the first argument.
#
#   (none)  sbatch runs this on the first node of the allocation. Archives a copy of this file
#           to $RUN_DIR, computes RAY_ADDRESS, starts one `node` task per node with srun.
#   node    one task per node, on the host (outside the container). Asserts the GPU count,
#           stages the checkpoint to /tmp, enters the container and runs `inner`.
#   inner   inside the container. Applies the environment (section 3), runs `ray symmetric-run`,
#           which elects head/workers itself and runs `train` on the head once every node joined.
#   train   the driver, on the head node, inside the container, under ray. Builds the SkyRL
#           arguments (section 2) and runs the entrypoint.
#
# The environment (section 3) is a function, so it can also be loaded into an interactive
# shell inside the container for debugging:
#     singularity exec -B "$BINDS" "$SANDBOX" bash
#     source /path/to/skyrl.sh
#
# Submit:        sbatch skyrl.sh
# Scale:         POLICY_NODES=4 INFER_NODES=4 sbatch --nodes=8 skyrl.sh
# Override env:  EXTRA_ENV="NCCL_NET=Socket NCCL_DEBUG=WARN" sbatch skyrl.sh
# Every variable in sections 1 and 2 uses ": ${VAR:=default}", so it can be overridden the same
# way, exported before sbatch (sbatch passes the submitting shell's environment into the job).
#
# Sections: 1 run config (edit for every experiment), 2 site config (paths, rarely changes),
# 3 training args, 4 environment, 5 launch plumbing, 6 dispatch. Editing 5 and 6 should be rare.
#
# Both config functions run in every role (batch, node, inner, train), each in its own process,
# so the values are recomputed everywhere. `set -a` exports every assignment in the block, so
# the values also reach the Python actors without listing them.

# =====================================================================
# 1. RUN CONFIG                                              
# =====================================================================
run_config() {
  set -a

  # cluster
  : "${NUM_NODES:=${SLURM_JOB_NUM_NODES:-4}}"
  : "${GPUS_PER_NODE:=8}"
  : "${POLICY_NODES:=2}"
  : "${INFER_NODES:=2}"
  : "${REF_NODES:=0}"                          # POLICY_NODES + INFER_NODES + REF_NODES == NUM_NODES

  # parallelism
  : "${TP:=4}"                                 # vLLM tensor parallel per engine; must divide GPUS_PER_NODE
  : "${SP:=4}"                                 # Ulysses sequence parallel for the policy; must divide policy GPUs

  # model
  : "${MODEL_SRC:=$ROOT/tildeopen}"            # local dir or hub id (HF_HUB_OFFLINE=1 is set, so pre-download)
  : "${SERVED_NAME:=$(basename "$MODEL_SRC")}" # no '/' allowed: Harbor addresses hosted_vllm/<name>
  # MODEL (the staged /tmp copy) is set by the `node` role, not here.

  # context
  : "${MAX_MODEL_LEN_CAP:=65536}"              # final MAX_MODEL_LEN = min(cap, max_position_embeddings from config.json)
  : "${MAX_GEN_LEN:=2048}"                     # per LLM call (one agent turn), not per trajectory
  : "${MAX_TURNS:=2}"

  # rollout
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
  : "${PROJECT_NAME:=gsm8k}"

  set +a
}

# =====================================================================
# 2. SITE CONFIG (paths, container, network; loaded before section 1)
# =====================================================================
site_config() {
  set -a
  : "${ROOT:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor}"
  : "${BINDS:=/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038}"
  : "${SANDBOX:=$ROOT/multitorch_sandbox.sif}"
  : "${DATA_DIR:=$ROOT/data}"                  # train.parquet and validation.parquet
  : "${CHAT_TEMPLATE:=/opt/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"
  : "${LOG_DIR:=$ROOT/logs}"
  : "${CKPT_DIR:=$ROOT/ckpts}"
  : "${RUNS_DIR:=$ROOT/previous_runs}"
  : "${RAY_PORT:=6379}"
  set +a
}

# =====================================================================
# 3. TRAINING ARGS (role: train, driver on the head node)
# =====================================================================
die() { echo "FATAL: $*" >&2; exit 1; }

role_train() {
  set -x

  # --- derived ---
  POLICY_GPUS=$(( POLICY_NODES * GPUS_PER_NODE ))
  REF_GPUS=$(( REF_NODES * GPUS_PER_NODE ))
  INFER_GPUS=$(( INFER_NODES * GPUS_PER_NODE ))
  NUM_ENGINES=$(( INFER_GPUS / TP ))           # vLLM replicas
  DP=$(( POLICY_GPUS / SP ))                   # FSDP data parallel ranks

  MODEL_CTX=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["max_position_embeddings"])' "$MODEL" 2>/dev/null || echo 32768)
  MAX_MODEL_LEN=$(( MODEL_CTX < MAX_MODEL_LEN_CAP ? MODEL_CTX : MAX_MODEL_LEN_CAP ))

  # layout
  (( POLICY_NODES >= 1 && INFER_NODES >= 1 ))                 || die "need at least one policy node and one inference node"
  (( POLICY_NODES + INFER_NODES + REF_NODES == NUM_NODES ))   || die "POLICY_NODES ($POLICY_NODES) + INFER_NODES ($INFER_NODES) + REF_NODES ($REF_NODES) != NUM_NODES ($NUM_NODES)"
  (( GPUS_PER_NODE % TP == 0 ))                               || die "TP=$TP must divide GPUS_PER_NODE=$GPUS_PER_NODE (an engine never spans nodes)"
  (( POLICY_GPUS % SP == 0 ))                                 || die "SP=$SP must divide policy GPUs=$POLICY_GPUS"
  (( REF_GPUS % SP == 0 ))                                    || die "SP=$SP must divide ref GPUs=$REF_GPUS"
  (( DP >= 2 ))                                               || die "DP=$DP: lower SP or raise POLICY_NODES"

  # batch rule (skyrl/train/utils/utils.py): MINI_BS * N_SAMPLES / DP must be an integer divisible by MICRO_TRAIN_BS
  (( TRAIN_BS % MINI_BS == 0 ))                               || die "TRAIN_BS=$TRAIN_BS must be a multiple of MINI_BS=$MINI_BS"
  (( MINI_BS * N_SAMPLES % DP == 0 ))                         || die "MINI_BS*N_SAMPLES=$(( MINI_BS * N_SAMPLES )) not divisible by DP=$DP"
  (( MINI_BS * N_SAMPLES / DP % MICRO_TRAIN_BS == 0 ))        || die "per-rank batch $(( MINI_BS * N_SAMPLES / DP )) not divisible by MICRO_TRAIN_BS=$MICRO_TRAIN_BS"

  # context
  (( MAX_GEN_LEN < MAX_MODEL_LEN ))                           || die "MAX_GEN_LEN=$MAX_GEN_LEN must be below MAX_MODEL_LEN=$MAX_MODEL_LEN"

  echo "layout: ${NUM_NODES}n | policy ${POLICY_NODES}n/${POLICY_GPUS} GCDs SP=$SP DP=$DP | vLLM ${INFER_NODES}n/${INFER_GPUS} GCDs TP=$TP x $NUM_ENGINES engines | ctx $MAX_MODEL_LEN | $TRAIN_BS tasks x $N_SAMPLES samples, $(( MINI_BS * N_SAMPLES / DP )) per rank"

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

    trainer.placement.colocate_all=false
    # an option to share the gpus for inference/policy/critics etc. (WE DON'T USE colocate_all=true)

    trainer.strategy=fsdp
    trainer.policy.fsdp_config.fsdp_size=16
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
    # how many DISTINCT tasks for each inference step
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
    trainer.project_name="$PROJECT_NAME"
    trainer.run_name="${PROJECT_NAME}_lumi_${NUM_NODES}n_${SERVED_NAME}_sp${SP}_dp${DP}"
    trainer.resume_mode=null
    trainer.log_path="$LOG_DIR"
    trainer.ckpt_path="$CKPT_DIR/${PROJECT_NAME}_${SERVED_NAME}_ckpt"
  )

  python -m skyrl.train.entrypoints.main_base "${ARGS[@]}" "$@"
}

# =====================================================================
# 4. ENVIRONMENT (applied inside the container by every Ray node and the driver)
# =====================================================================
# Each raylet has its own environment and SkyRL/vLLM actors inherit it, so every variable
# an actor must see lives here, not in the batch role or the driver.
#
#   4a. common settings (always on)
#   4b. transport: Libfabric   <- comment out exactly one of the two transport blocks
#   4c. transport: Socket
#   4d. debug / logging and per-run overrides
apply_env() {
  # ------------------------------------------------ 4a. COMMON
  # --- Python / HF ---
  . /opt/venv/bin/activate
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="$ROOT/pyhooks${PYTHONPATH:+:$PYTHONPATH}" # is not used
  export HF_HOME="$ROOT/hf"
  export HF_HUB_OFFLINE=1
  export HF_ENABLE_PARALLEL_LOADING=true     # rank 0 reads the safetensors shards concurrently
  export HF_PARALLEL_LOADING_WORKERS=8

  # --- GPU visibility ---
  # Slurm sets ROCR_VISIBLE_DEVICES; Ray refuses it and wants HIP_VISIBLE_DEVICES.
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export HWLOC_COMPONENTS=-amd_smi,-rsmi

  # --- node-local caches (/tmp is RAM, per node, charged to the job's memory cgroup) ---
  export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-${SLURM_JOB_ID:-x}"
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
  export TORCH_EXTENSIONS_DIR="/tmp/$USER-torch-ext-${SLURM_JOB_ID:-x}"
  mkdir -p "$MIOPEN_USER_DB_PATH" "$TORCH_EXTENSIONS_DIR"

  # --- Ray ---
  export RAY_USAGE_STATS_ENABLED=0
  export RAY_DEDUP_LOGS=0
  # symmetric-run defaults are 30 s; nodes here start staggered (checkpoint staging, container, venv).
  export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=300
  export RAY_py_gcs_connect_timeout_s=300

  # --- SkyRL / vLLM ---
  export SKYRL_LD_LIBRARY_PATH_EXPORT=True   # forward LD_LIBRARY_PATH to Ray workers
  # Worker process-group timeout. Rank 0 alone loads the HF checkpoint (bf16 -> fp32 in host RAM)
  # while the other ranks wait in the first gloo broadcast; TildeOpen-30b takes ~10 min from /scratch.
  export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export OMP_NUM_THREADS=7

  # --- network settings shared by both transports ---
  # Bootstrap/OOB traffic on the Slingshot NICs. Under Socket this list is also the data path,
  # so keep it identical across benchmark runs.
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export GLOO_SOCKET_IFNAME=hsn0
  export CXI_FORK_SAFE=1
  export CXI_FORK_SAFE_HP=1
  export RCCL_MSCCL_ENABLE=0
  # Weight-sync communicator is 1 trainer GCD + 8 GCDs per vLLM node. Without this the 8-GCD ranks
  # match RCCL's MI250X topology model and build an extra connection set, the 1-GCD rank does not,
  # and ncclCommInitRank deadlocks (job 22820955). Must be set on every rank.
  export RCCL_MODEL_MATCHING_DISABLE=1

  # ------------------------------------------------ 4b. TRANSPORT: LIBFABRIC
  # (aws-ofi-nccl 1.20.0 plugin on libfabric 2.1, cxi provider)
  export NCCL_NET=Libfabric
  # Some RCCL versions look for libnccl-net.so instead of librccl-net.so; pin the path when present.
  if [ -f /usr/lib/x86_64-linux-gnu/librccl-net.so ]; then
    export NCCL_NET_PLUGIN=/usr/lib/x86_64-linux-gnu/librccl-net.so
  fi
  # GPU Direct RDMA level (LUMI docs and HPE guide recommend PHB). LOC = host-staged.
  # Never set NCCL_DMABUF_ENABLE here: RCCL then reads /boot/config-*, which is not in the
  # container, and segfaults in ncclCommInitRank (job 22817181). The plugin detects DMA-BUF itself.
  export NCCL_NET_GDR_LEVEL=PHB
  # libfabric/cxi provider settings (HPE RCCL tuning guide, LAIF image defaults).
  # kdreg2 is requested but this container's libfabric cannot use it ("kdreg2 monitor not
  # available" with FI_LOG_LEVEL=warn); libfabric falls back to another monitor. Harmless, noted.
  export FI_MR_CACHE_MONITOR=userfaultfd
  export FI_CXI_DISABLE_HOST_REGISTER=1
  export FI_CXI_DISABLE_CQ_HUGETLB=1
  # REQUIRED. Without it every all-reduce over Libfabric hangs while broadcasts pass (weight sync
  # hangs in PyNcclCommunicator.__init__, vLLM then reports shm_broadcast.py "No available shared
  # memory broadcast block"). Cause, from the stack of the stuck rank (job 22895140,
  # investigation/stacks-22895140/pystack.rank1.txt): for a small message the cxi provider builds
  # an immediate-data command (IDC) by copying the bytes out of the source buffer; that copy runs
  # through rocr_memcpy on the RCCL proxy thread and waits on the GPU, while the RCCL kernel on
  # that GPU waits for the send. Circular wait, no error. With IDC disabled the NIC reads the
  # registered buffer directly and no copy is made. Reproducer: scripts/submit_rccl_test.sbatch
  # (hang 22879416; pass 22896687, 22897075, 22897363). Reported to LUMI support.
  export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1

  # ------------------------------------------------ 4c. TRANSPORT: SOCKET
  # (TCP over the hsn interfaces listed in NCCL_SOCKET_IFNAME)
  # export NCCL_NET=Socket
  # Everything in 4b is inert under Socket (plugin pin, GDR level, FI_* variables), so the only
  # line that has to change between the two transports is NCCL_NET. For one-off runs use
  # EXTRA_ENV="NCCL_NET=Socket" instead of editing this file.

  # ------------------------------------------------ 4d. DEBUG / LOGGING / OVERRIDES
  # One RCCL log per process (%h host, %p pid). ENV prints the variables each rank read.
  # Check after every multi-node run: "Using network Libfabric" or "Using network Socket" as intended.
  # For benchmark runs use NCCL_DEBUG=WARN or drop COLL: at INFO, COLL logs every collective to Lustre.
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV,COLL}"
  export NCCL_DEBUG_TIMESTAMP_LEVELS=ALL
  export NCCL_DEBUG_FILE="$LOG_DIR/nccl-${SLURM_JOB_ID:-x}/%h.%p.log"
  mkdir -p "$LOG_DIR/nccl-${SLURM_JOB_ID:-x}"

  # Per-run overrides for experiments, applied last so they win over everything above:
  #   EXTRA_ENV="NCCL_NET=Socket NCCL_DEBUG=WARN" sbatch skyrl.sh
  # Values must not contain spaces. Echoed once per node in the job output.
  if [ -n "${EXTRA_ENV:-}" ]; then
    # shellcheck disable=SC2086
    export $EXTRA_ENV
    echo "[$(hostname)] EXTRA_ENV applied: $EXTRA_ENV"
  fi
}

# =====================================================================
# 5. LAUNCH PLUMBING (roles: batch, node, inner)
# =====================================================================

# --- role: batch. Runs under sbatch on the first node of the allocation. ---
role_batch() {
  # Run archive: snapshot this file now, copy the log when the job ends.
  #
  # --output=%x-%j.out is relative to the directory sbatch was run from, so the log lives at
  # $SLURM_SUBMIT_DIR/<jobname>-<jobid>.out. Slurm keeps appending to it for the whole job, so a
  # copy taken now would be nearly empty. Instead the copy is done in an EXIT trap, which runs
  # after srun returns, whether the job finished, failed (set -e), or was cancelled / timed out
  # (the TERM trap turns the signal into an exit so the EXIT trap still fires).
  RUN_DIR="$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
  LOG_FILE="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  mkdir -p "$RUN_DIR"

  # Slurm executes a private copy of the batch script from /var/spool/slurmd on this node only.
  # The other nodes and the container need the file on /scratch, so the archived copy is the one
  # every later role runs. Side effect: the run is frozen. Editing this file after submission
  # does not change a queued or running job.
  export SELF="$RUN_DIR/skyrl.sh"
  cp "${BASH_SOURCE[0]}" "$SELF"
  scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/slurm_job.txt"
  # Effective values of sections 1 and 2 (defaults plus any overrides from the submitting shell).
  { declare -f run_config site_config | grep -oE '\$\{[A-Z_]+:=' | tr -d '${:=' | sort -u \
      | while read -r v; do printf '%s=%s\n' "$v" "${!v:-}"; done
    if [ -n "${EXTRA_ENV:-}" ]; then echo "EXTRA_ENV=$EXTRA_ENV"; fi; } > "$RUN_DIR/config.txt"
  echo "run archive: $RUN_DIR"

  LATEST_DIR="$RUNS_DIR/latest_run"
  rm -rf "$LATEST_DIR"
  mkdir -p "$LATEST_DIR"
  for f in "$RUN_DIR"/*; do
    ln -s "$f" "$LATEST_DIR/$(basename "$f")"
  done
  ln -s "$LOG_FILE" "$LATEST_DIR/$(basename "$LOG_FILE")"
  ln -s "$RUN_DIR"  "$LATEST_DIR/run_dir"

  archive_log() {
    local rc=$?
    echo "job finished with exit code $rc, copying log to $RUN_DIR"
    cp -f "$LOG_FILE" "$RUN_DIR/" 2>/dev/null || true
    exit "$rc"
  }
  trap archive_log EXIT
  trap 'exit 143' TERM INT HUP

  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE="${NODES[0]}"

  # The batch script runs on the first node of the allocation, which is also our Ray head node.
  # Assert that so the local hsn0 lookup below can't silently be wrong.
  [ "${SLURMD_NODENAME:-$(hostname)}" = "$HEAD_NODE" ] || {
    echo "FATAL: batch host ${SLURMD_NODENAME:-$(hostname)} != first node $HEAD_NODE" >&2
    exit 1
  }

  # Ray control traffic on the Slingshot fabric, not the management network. symmetric-run
  # compares this address against every local interface to elect the head, and passes it to
  # `ray start --head --node-ip-address=...`.
  HEAD_IP=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
  export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
  echo "nodes: ${NODES[*]}   head: $HEAD_NODE ($RAY_ADDRESS)"

  # --label            : prefix every output line with the task rank, so it is immediately
  #                      visible which task printed "On head node" (must be exactly one).
  # --kill-on-bad-exit : any task failing (GPU assertion, ray start crash, timeout) kills the
  #                      whole step instead of leaving the others waiting.
  # --mem=0 matters on dev-g (OOM otherwise).
  srun --label --kill-on-bad-exit=1 --mem=0 \
       --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
       --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
       bash "$SELF" node
}

# --- role: node. One srun task per node, on the host, outside the container. ---
role_node() {
  # GPU assertion must happen here, on the host side, before apply_env unsets ROCR_VISIBLE_DEVICES
  # inside the container.
  local ngpu
  ngpu=$(tr ',' '\n' <<< "${ROCR_VISIBLE_DEVICES:-}" | grep -c .)
  [ "$ngpu" -eq "$GPUS_PER_NODE" ] || {
    echo "[$SLURMD_NODENAME] FATAL: step got $ngpu/$GPUS_PER_NODE GPUs (ROCR=${ROCR_VISIBLE_DEVICES:-unset})" >&2
    exit 1
  }

  # Diagnostic: the `src` address in this line is what a worker raylet will bind to
  # (ray start derives it from the route to the head). Expect an hsn* interface.
  echo "[$SLURMD_NODENAME] route to head: $(ip route get "${RAY_ADDRESS%%:*}" | head -1)"

  # Stage the checkpoint into node-local RAM (/tmp): sequential reads instead of mmap page faults on Lustre.
  local src
  if [ -d "$MODEL_SRC" ]; then
    src="${MODEL_SRC%/}"
  else
    src=$(ls -d "$ROOT/hf/hub/models--${MODEL_SRC//\//--}"/snapshots/*/ | head -1)
    src="${src%/}"
  fi
  export MODEL="/tmp/$USER-model/$(basename "$MODEL_SRC")"
  mkdir -p "$MODEL"
  echo "[$SLURMD_NODENAME] staging $src -> $MODEL, timing on the next line:"
  TIMEFORMAT="[$SLURMD_NODENAME] staging took %R s"
  time (ls "$src" | xargs -P 16 -I{} cp -rL "$src/{}" "$MODEL/")

  exec singularity exec -B "$BINDS" "$SANDBOX" bash "$SELF" inner
}

# --- role: inner. Inside the container, one per node. ---
role_inner() {
  apply_env
  ray stop --force >/dev/null 2>&1 || true
  # Not allowed with symmetric-run: --head, --node-ip-address, --port, --block. It adds those
  # itself. Everything else is forwarded to `ray start` on every node. The command after `--`
  # runs on the head node only, once --min-nodes have joined.
  exec ray symmetric-run \
    --address "$RAY_ADDRESS" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus="$SLURM_CPUS_PER_TASK" \
    --num-gpus="$GPUS_PER_NODE" \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    -- bash "$SELF" train
}

# =====================================================================
# 6. DISPATCH
# =====================================================================
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # Sourced into an interactive shell (debugging inside the container): load config and
  # environment, do not run anything, do not change the caller's shell options.
  site_config
  run_config
  apply_env
  return 0
fi

set -euo pipefail
site_config
run_config

case "${1:-batch}" in
  batch) role_batch ;;
  node)  role_node ;;
  inner) role_inner ;;
  train) shift; role_train "$@" ;;
  *)     echo "usage: sbatch $0   (or: bash $0 {node|inner|train}, or: source $0)" >&2; exit 2 ;;
esac