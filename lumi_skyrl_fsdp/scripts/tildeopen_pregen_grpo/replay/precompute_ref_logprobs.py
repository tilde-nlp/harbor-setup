#!/usr/bin/env python
"""
Add a `ref_logprobs` column to the replay parquet: log pi_ref(token) for every response token,
computed once with vLLM prefill (prompt_logprobs) from the checkpoint that generated the data.

Resumable and shardable:
  * --shard i --num-shards n   this process scores rows i, i+n, i+2n, ... (run n processes on
                               disjoint GPU sets, e.g. two engines with TP=4 on one node)
  * every batch is written as its own parquet file under <out>.d/shard<i>/batch_<start>.parquet
    (temp name, then rename), so a killed job loses at most one batch; existing batch files are
    skipped on restart
  * --merge                    assemble <out> from all batch files, in original row order, and
                               verify that every row of the input is present exactly once

vLLM runs with skip_tokenizer_init=True (inputs and outputs are token ids; the model dir only has
a slow sentencepiece tokenizer and the container has no `sentencepiece`).

Typical use is replay/submit_precompute_ref_logprobs.sbatch. By hand:

    HIP_VISIBLE_DEVICES=0,1,2,3 python -m replay.precompute_ref_logprobs --parquet in.parquet --out out.parquet \
        --model $MODEL_SRC --tp 4 --shard 0 --num-shards 2 &
    HIP_VISIBLE_DEVICES=4,5,6,7 python -m replay.precompute_ref_logprobs --parquet in.parquet --out out.parquet \
        --model $MODEL_SRC --tp 4 --shard 1 --num-shards 2 &
    wait
    python -m replay.precompute_ref_logprobs --parquet in.parquet --out out.parquet --merge

Notes
  * Sequences are prompt_ids + response_ids. vLLM needs room for one generated token, so rows
    with len > max_model_len - 1 are refused unless --drop-too-long.
  * prompt_logprobs=0 returns only the log-prob of the actual token at each position (raw model
    logits, temperature 1.0, which is what the trainer uses with sampling_params.temperature=1.0).
  * prompt_logprobs materialises chunk x vocab x 4 bytes twice; keep --max-num-batched-tokens at
    8192 or less with a 131k vocabulary, and leave GPU headroom (--gpu-memory-utilization 0.7).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def batch_dir(out, shard):
    return f"{out}.d/shard{shard}"


def load_input(path, limit):
    table = pq.read_table(path)
    if limit:
        table = table.slice(0, limit)
    if "ref_logprobs" in table.column_names:
        table = table.drop_columns(["ref_logprobs"])
    if "_row" in table.column_names:
        table = table.drop_columns(["_row"])
    return table.append_column("_row", pa.array(np.arange(table.num_rows, dtype=np.int64)))


def score(args):
    from tqdm import tqdm

    table = load_input(args.parquet, args.limit)
    prompt_ids = table.column("prompt_ids").to_pylist()
    response_ids = table.column("response_ids").to_pylist()
    lens = np.array([len(p) + len(r) for p, r in zip(prompt_ids, response_ids)])
    too_long = {i for i, L in enumerate(lens) if L > args.max_model_len - 1}
    if too_long and not args.drop_too_long:
        print(f"{len(too_long)} rows exceed max_model_len-1={args.max_model_len - 1}; raise --max-model-len or pass --drop-too-long", file=sys.stderr)
        sys.exit(1)
    mine = [i for i in range(table.num_rows) if i % args.num_shards == args.shard and i not in too_long]
    batches = [mine[s : s + args.batch_rows] for s in range(0, len(mine), args.batch_rows)]
    d = batch_dir(args.out, args.shard)
    os.makedirs(d, exist_ok=True)
    todo = [(b_i, b) for b_i, b in enumerate(batches) if not os.path.exists(f"{d}/batch_{b_i:06d}.parquet")]
    done_tokens = int(sum(lens[b].sum() for b_i, b in enumerate(batches) if (b_i, b) not in todo))
    total_tokens = int(lens[mine].sum())
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} rows, {total_tokens / 1e6:.1f}M tokens, "
          f"{len(batches) - len(todo)}/{len(batches)} batches already done ({done_tokens / 1e6:.1f}M tokens)", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens, gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True, max_logprobs=1,
    )
    if args.tokenizer:
        llm_kwargs["tokenizer"] = args.tokenizer
    else:
        llm_kwargs["skip_tokenizer_init"] = True
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)
    schema = table.schema.append(pa.field("ref_logprobs", pa.list_(pa.float32())))

    bar = tqdm(total=total_tokens, initial=done_tokens, unit="tok", unit_scale=True,
               desc=f"shard {args.shard}", dynamic_ncols=True, mininterval=10.0)
    t0, t0_tokens = time.time(), done_tokens
    for b_i, idx in todo:
        prompts = [{"prompt_token_ids": prompt_ids[i] + response_ids[i]} for i in idx]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        col = []
        for i, out in zip(idx, outs):
            p_len, resp = len(prompt_ids[i]), response_ids[i]
            plp = out.prompt_logprobs  # one entry per input position; entry 0 is None
            if plp is None or len(plp) != p_len + len(resp):
                raise RuntimeError(f"row {i}: got {None if plp is None else len(plp)} prompt logprobs for {p_len + len(resp)} tokens")
            lp = np.empty(len(resp), dtype=np.float32)
            for k, tok in enumerate(resp):
                entry = plp[p_len + k]
                if entry is None or tok not in entry:
                    raise RuntimeError(f"row {i}: missing logprob for response token {k} (id {tok})")
                lp[k] = entry[tok].logprob
            col.append(lp.tolist())
        batch = table.take(pa.array(idx)).append_column("ref_logprobs", pa.array(col, type=pa.list_(pa.float32())))
        tmp = f"{d}/batch_{b_i:06d}.parquet.tmp"
        pq.write_table(batch, tmp, compression="zstd")
        os.replace(tmp, f"{d}/batch_{b_i:06d}.parquet")
        done_tokens += int(lens[idx].sum())
        bar.update(int(lens[idx].sum()))
        bar.set_postfix(batch=f"{b_i + 1}/{len(batches)}", rate=f"{(done_tokens - t0_tokens) / (time.time() - t0) / 1e3:.1f}k tok/s")
    bar.close()
    print(f"shard {args.shard}: done, {len(batches)} batches in {d}", flush=True)


def merge(args):
    table = load_input(args.parquet, args.limit)
    files = sorted(glob.glob(f"{args.out}.d/shard*/batch_*.parquet"))
    if not files:
        print(f"no batch files under {args.out}.d", file=sys.stderr)
        sys.exit(1)
    parts = pq.ParquetDataset(files).read()
    rows = parts.column("_row").to_numpy()
    expected = table.num_rows
    missing = np.setdiff1d(np.arange(expected), rows)
    if len(rows) != len(np.unique(rows)):
        print("duplicate rows across batch files", file=sys.stderr)
        sys.exit(1)
    if len(missing) and not args.allow_missing:
        print(f"{len(missing)} of {expected} rows not scored yet (e.g. {missing[:5].tolist()}); rerun the scoring, "
              f"or pass --allow-missing to merge what exists", file=sys.stderr)
        sys.exit(1)
    merged = parts.take(pa.array(np.argsort(rows, kind="stable"))).drop_columns(["_row"])
    lp = merged.column("ref_logprobs")
    n_ref = pa.compute.list_value_length(lp).to_numpy()
    n_resp = pa.compute.list_value_length(merged.column("response_ids")).to_numpy()
    assert (n_ref == n_resp).all(), "ref_logprobs length != response_ids length in some rows"
    flat = pa.compute.list_flatten(lp).to_numpy()
    tmp = args.out + ".tmp"
    pq.write_table(merged, tmp, compression="zstd")
    os.replace(tmp, args.out)
    print(f"merged {merged.num_rows} rows ({len(files)} batch files) -> {args.out}")
    print(f"ref_logprobs: {len(flat)} values, mean {flat.mean():.3f}, min {flat.min():.2f}, "
          f"frac near-certain (> -1e-4) {(flat > -1e-4).mean():.3f}")
    if len(missing):
        print(f"WARNING: {len(missing)} rows missing (--allow-missing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--tokenizer", default=None, help="only if vLLM insists on a tokenizer; a dir with tokenizer.json")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--batch-rows", type=int, default=256, help="rows per batch file")
    ap.add_argument("--drop-too-long", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge(args)
    else:
        assert args.model, "--model is required for scoring"
        score(args)


if __name__ == "__main__":
    main()