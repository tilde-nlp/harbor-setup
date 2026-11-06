#!/usr/bin/env python
"""
Build the ref_logprobs column from the training stack's own forward pass.

Input: the dumps written by a REPLAY_SCORE_ONLY=1 pass (trainer.dump_data_batch=true,
policy_loss_type=regular): <export_path>/dumped_data/global_step_<n>_training_input.pkl, each
holding `sequences` (left-padded prompt+response), `loss_mask`/`response_mask` (right-aligned)
and `action_log_probs` (the policy worker's log-probs of the response tokens, no grad).

Every dumped sequence is matched to its parquet row by the full token content (prompt_ids +
response_ids); identical rows (same prompt, same rollout) get the same log-probs, which is
correct since the model sees the same context. The output parquet has every input row, in the
input order, with `ref_logprobs` replaced by the trainer's values. If the dumps also carried
rollout_logprobs (the vLLM column), the script reports how far the two stacks are apart.

    python -m replay.collect_trainer_logprobs \
        --dumps $ROOT/exports/score/dumped_data --parquet replay.ref.parquet --out replay.ref_trainer.parquet
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def find_tensors(obj, depth=0):
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from find_tensors(v, depth + 1) if not (hasattr(v, "shape") and hasattr(v, "dtype")) else [(k, to_np(v))]
        return
    for attr in ("batch", "tensors", "data", "__dict__"):
        sub = getattr(obj, attr, None)
        if sub is not None and sub is not obj:
            yield from find_tensors(sub, depth + 1)
            return
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            for k in obj.keys():
                v = obj[k]
                if hasattr(v, "shape") and hasattr(v, "dtype"):
                    yield k, to_np(v)
                else:
                    yield from find_tensors(v, depth + 1)
        except Exception:
            pass


def load_dump(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="directory with global_step_*_training_input.pkl")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true", help="write rows without a score as nulls instead of failing")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dumps, "global_step_*_training_input.pkl")),
                   key=lambda f: int(re.search(r"global_step_(\d+)_", f).group(1)))
    if not files:
        print(f"no dumps in {args.dumps}", file=sys.stderr)
        sys.exit(1)

    table = pq.read_table(args.parquet)
    prompt_ids = table.column("prompt_ids").to_pylist()
    response_ids = table.column("response_ids").to_pylist()
    index = {}
    for i, (p, r) in enumerate(zip(prompt_ids, response_ids)):
        index.setdefault((tuple(p[-4:]), tuple(r[-4:]), len(p), len(r)), []).append(i)

    scores = [None] * table.num_rows
    n_seq = n_matched = 0
    diffs = []
    for fi, f in enumerate(files):
        t = dict(find_tensors(load_dump(f)))
        seq, al = t["sequences"], t["action_log_probs"]
        rm = t.get("response_mask", (t["loss_mask"] != 0) | (t["loss_mask"] == 0))  # fallback: infer from attention
        am = t.get("attention_mask")
        B, T = seq.shape
        R = al.shape[1]
        for b in range(B):
            n_seq += 1
            row = seq[b].tolist()
            L_resp = int(rm[b].sum()) if "response_mask" in t else None
            # total non-pad length from attention_mask; prompt length = total - response
            L_tot = int(am[b].sum()) if am is not None else None
            if L_resp is None or L_tot is None:
                print(f"{f}: cannot determine lengths for seq {b}", file=sys.stderr)
                sys.exit(1)
            L_p = L_tot - L_resp
            p_tok, r_tok = row[T - L_tot : T - L_resp], row[T - L_resp :]
            cands = index.get((tuple(p_tok[-4:]), tuple(r_tok[-4:]), L_p, L_resp), [])
            hit = [i for i in cands if prompt_ids[i] == p_tok and response_ids[i] == r_tok]
            if not hit:
                print(f"{f}: seq {b} (prompt {L_p}, response {L_resp}) not found in the parquet", file=sys.stderr)
                continue
            lp = al[b, R - L_resp :].astype(np.float32)
            for i in hit:
                if scores[i] is None:
                    scores[i] = lp
                    n_matched += 1
            if "rollout_logprobs" in t:
                m = t["loss_mask"][b, R - L_resp :] > 0
                diffs.append(np.abs(lp - t["rollout_logprobs"][b, R - L_resp :])[m])
        if (fi + 1) % 50 == 0 or fi + 1 == len(files):
            print(f"{fi + 1}/{len(files)} dumps, {n_matched}/{table.num_rows} rows scored", flush=True)

    missing = [i for i, s in enumerate(scores) if s is None]
    print(f"dumped sequences {n_seq}, rows scored {n_matched}/{table.num_rows}, missing {len(missing)}")
    if missing and not args.allow_missing:
        print(f"missing rows e.g. {missing[:5]}; the scoring pass did not cover the whole dataset "
              f"(epochs=1 and every batch dumped?), or pass --allow-missing", file=sys.stderr)
        sys.exit(1)
    if diffs:
        d = np.concatenate(diffs)
        print(f"trainer vs vLLM on loss tokens: mean |diff| {d.mean():.4f}, p90 {np.percentile(d, 90):.3f}, "
              f"frac > 0.2: {(d > 0.2).mean():.3f}")

    col = pa.array([None if s is None else s.tolist() for s in scores], type=pa.list_(pa.float32()))
    out = table.drop_columns(["ref_logprobs"]) if "ref_logprobs" in table.column_names else table
    out = out.append_column("ref_logprobs", col)
    tmp = args.out + ".tmp"
    pq.write_table(out, tmp, compression="zstd")
    os.replace(tmp, args.out)
    flat = np.concatenate([s for s in scores if s is not None])
    print(f"wrote {out.num_rows} rows -> {args.out}; ref_logprobs mean {flat.mean():.3f}, min {flat.min():.2f}")


if __name__ == "__main__":
    main()