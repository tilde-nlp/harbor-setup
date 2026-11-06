#!/bin/bash
#SBATCH --job-name=ref-logprobs
#SBATCH --account=project_465003062
#SBATCH --partition=standard-g         # dev-g caps at 3 h; the first attempt ran 8k tok/s = ~2.4 h on 8 GCDs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# One-time job: add ref_logprobs (log-probs of the data-generating checkpoint) to the replay
# parquet with vLLM prefill on one node. No Ray, no training stack.
#
# Submit:   sbatch submit_precompute_ref_logprobs.sbatch      (resubmit the same way after a timeout: it resumes)
# Override: PARQUET_IN=... PARQUET_OUT=... TP=8 MAX_MODEL_LEN=49152 sbatch submit_precompute_ref_logprobs.sbatch
#
# The environment (paths, container, venv, GPU visibility, RCCL settings) comes from sourcing the
# training sbatch inside the container, so the two jobs cannot drift apart.

set -euo pipefail

: "${ROOT:=/scratch/project_465003062/environment/containers/grpo-pregen-rl}"
: "${TRAIN_SBATCH:=$ROOT/launch_scripts/submit_tildeopen_pregen_grpo_4nodes.sbatch}"
: "${PARQUET_IN:=$ROOT/data/replay/replay.parquet}"
: "${PARQUET_OUT:=$ROOT/data/replay/replay.ref.parquet}"
: "${TP:=4}"                    # per engine
: "${NUM_SHARDS:=2}"            # engines run in parallel on disjoint GCD sets (TP*NUM_SHARDS = 8)
: "${BATCH_ROWS:=256}"          # rows per batch file; a kill loses at most one batch
: "${MAX_MODEL_LEN:=49152}"     # >= max prompt+response in the parquet + 1 (47048 + 1 for this dataset)
: "${MAX_NUM_BATCHED_TOKENS:=8192}"  # prefill chunk. prompt_logprobs materialises chunk x vocab (131k) x 4 B twice:
                                    # 8192 -> ~8.6 GiB, 32768 -> ~34 GiB (OOM, job 23176047)
: "${GPU_MEM_UTIL:=0.70}"           # KV-cache share; the rest is headroom for those logprob buffers
: "${EXTRA:=}"                  # e.g. EXTRA="--limit 64" for a quick API check
: "${NET:=Socket}"              # transport for the intra-node TP group. Single node: no fabric needed.
                                # NET=Libfabric keeps the training sbatch's settings (plugin pin, FI_* vars).
: "${NCCL_DEBUG_LEVEL:=WARN}"   # NCCL log level for this job (training default INFO is very chatty)

export ROOT TRAIN_SBATCH PARQUET_IN PARQUET_OUT TP NUM_SHARDS BATCH_ROWS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS GPU_MEM_UTIL EXTRA NET NCCL_DEBUG_LEVEL

# site config of the training sbatch gives BINDS and SANDBOX; load it on the host side only for those
# shellcheck disable=SC1090
eval "$(sed -n '/^site_config() {/,/^}/p' "$TRAIN_SBATCH")"; site_config

echo "node $(hostname), GPUs ${ROCR_VISIBLE_DEVICES:-unset}, in=$PARQUET_IN out=$PARQUET_OUT"

srun --ntasks=1 --gpus-per-node=8 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem=0 \
  singularity exec -B "$BINDS" "$SANDBOX" bash -c '
    set -euo pipefail
    source "$TRAIN_SBATCH"                       # sourced: site_config + run_config + apply_env, runs nothing
    # single-node overrides, applied after apply_env so they win
    export NCCL_NET="$NET"
    if [ "$NET" = Socket ]; then unset NCCL_NET_PLUGIN NCCL_NET_GDR_LEVEL; fi
    export NCCL_DEBUG="$NCCL_DEBUG_LEVEL"
    echo "NCCL_NET=$NCCL_NET NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-unset} NCCL_DEBUG=$NCCL_DEBUG -> $NCCL_DEBUG_FILE"
    cd "$ROOT"
    echo "model $MODEL_SRC  tp $TP  max_model_len $MAX_MODEL_LEN"
    # one engine per shard on its own GCDs; batch files land in $PARQUET_OUT.d/shard<i>/ and are
    # skipped on a rerun, so resubmitting after a timeout continues where it stopped
    all_gpus=(${HIP_VISIBLE_DEVICES//,/ })
    pids=()
    for ((s=0; s<NUM_SHARDS; s++)); do
      gpus=$(IFS=,; echo "${all_gpus[*]:s*TP:TP}")
      echo "shard $s on GCDs $gpus"
      HIP_VISIBLE_DEVICES=$gpus python -m replay.precompute_ref_logprobs \
          --parquet "$PARQUET_IN" --out "$PARQUET_OUT" \
          --model "$MODEL_SRC" --tp "$TP" --shard "$s" --num-shards "$NUM_SHARDS" --batch-rows "$BATCH_ROWS" \
          --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
          --gpu-memory-utilization "$GPU_MEM_UTIL" $EXTRA 2>&1 | sed "s/^/[shard $s] /" &
      pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ $rc -eq 0 ] || { echo "a shard failed; resubmit to continue (done batches are kept)"; exit 1; }
    python -m replay.precompute_ref_logprobs --parquet "$PARQUET_IN" --out "$PARQUET_OUT" --merge $EXTRA
  '
