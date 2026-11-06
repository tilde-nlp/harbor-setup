#!/usr/bin/env python
"""
Check that the batch SkyRL hands to the policy worker carries the masks from the parquet.

Run one step with `trainer.dump_data_batch=true`; the trainer pickles the training input of
each step at  <trainer.export_path>/dumped_data/global_step_<N>_training_input.pkl
(trainer.py: dump_data(), called in train() after advantages are computed). Then:

    python -m replay.inspect_dump --dump <that .pkl> --parquet replay.parquet

Layout in SkyRL (skyrl/train/dataset/preprocess.py, convert_prompts_responses_to_batch_tensors):
each row of `sequences` is LEFT-padded: [pad...][prompt][response]. `response_mask`, `loss_mask`,
`advantages` and `rollout_logprobs` are (B, max_response) and RIGHT-aligned: a response of
length L occupies the last L columns, the leading columns are 0.

For every sequence in the dump this script checks:
  * the last L tokens of the row equal a stored rollout's response_ids (matched by content),
  * loss_mask on those columns equals the stored loss_mask and the leading columns are 0,
  * response_mask is 1 exactly on those columns,
  * advantages are 0 on the leading (padding) columns and constant over the response
    (GRPO outcome advantage: one scalar per sequence, broadcast over all response tokens,
    multiplied by response_mask, NOT by loss_mask; the tool-output mask is applied in the
    loss: reduce_loss(loss, loss_mask), compute_approx_kl(..., loss_mask), masked_mean(entropy, loss_mask)),
  * rollout_logprobs, if present, has the same shape as loss_mask.

Exit 1 if any check fails or no sequence could be matched. The pickle layout differs between
SkyRL versions; the loader searches tensors by name and prints every tensor it finds.
"""
import argparse
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

WANT = ["sequences", "attention_mask", "loss_mask", "response_mask", "advantages", "returns",
        "rollout_logprobs", "base_action_log_probs", "action_log_probs", "rewards"]


def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def find_tensors(obj, depth=0, prefix=""):
    """Yield (name, array) for every tensor-like leaf reachable from obj."""
    if depth > 4:
        return
    sep = "." if prefix else ""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from find_tensors(v, depth + 1, f"{prefix}{sep}{k}")
        return
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        yield prefix, to_np(obj)
        return
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(e, (int, float)) for e in obj):
            yield prefix, np.asarray(obj)
        return
    for attr in ("batch", "tensors", "data", "__dict__"):
        sub = getattr(obj, attr, None)
        if sub is not None and sub is not obj:
            yield from find_tensors(sub, depth + 1, f"{prefix}{sep}{attr}")
            return
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            for k in obj.keys():
                yield from find_tensors(obj[k], depth + 1, f"{prefix}{sep}{k}")
        except Exception:
            pass


def load_dump(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # TrainingInputBatch.save may use torch.save
        import torch
        print(f"pickle.load failed ({e}); trying torch.load", file=sys.stderr)
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-print", type=int, default=20)
    args = ap.parse_args()

    obj = load_dump(args.dump)
    found = {}
    print("tensors in dump:")
    for name, arr in find_tensors(obj):
        short = name.split(".")[-1]
        print(f"  {name:48s} {arr.shape} {arr.dtype}")
        if short in WANT and short not in found:
            found[short] = arr
    missing = [k for k in ("sequences", "loss_mask", "advantages") if k not in found]
    if missing:
        print(f"could not locate {missing}; map the names above and adjust WANT", file=sys.stderr)
        sys.exit(2)

    seq, lm, adv = found["sequences"], found["loss_mask"], found["advantages"]
    rm = found.get("response_mask")
    B, T = seq.shape
    R = lm.shape[1]
    print(f"\nbatch: {B} sequences, row length {T}, max response length {R}")

    # index stored rollouts by their LAST 4 response ids (responses are right-aligned)
    table = pq.read_table(args.parquet, columns=["uid", "rep", "response_ids", "loss_mask", "score"])
    store = {}
    for r in table.to_pylist():
        store.setdefault(tuple(r["response_ids"][-4:]), []).append(r)

    n_ok = n_unmatched = 0
    problems = []
    printed = 0

    def report(msg):
        nonlocal printed
        problems.append(msg)
        if printed < args.max_print:
            print(msg)
            printed += 1

    for b in range(B):
        row = seq[b].tolist()
        match = None
        for cand in store.get(tuple(row[-4:]), []):
            L = len(cand["response_ids"])
            if row[T - L:] == cand["response_ids"]:
                match = cand
                break
        if match is None:
            n_unmatched += 1
            report(f"UNMATCHED seq {b}: last ids {row[-6:]} not found in the parquet")
            continue
        L = len(match["response_ids"])
        tag = f"seq {b} uid={match['uid']} rep={match['rep']} L={L}"
        ok = True

        got = lm[b, R - L:].astype(int).tolist()
        exp = [int(m) for m in match["loss_mask"]]
        if got != exp:
            report(f"LOSS_MASK MISMATCH {tag}: {sum(got)} vs {sum(exp)} loss tokens, "
                   f"first diff at {next(i for i in range(L) if got[i] != exp[i])}")
            ok = False
        if lm[b, :R - L].any():
            report(f"LOSS_MASK PADDING NONZERO {tag}")
            ok = False
        if rm is not None:
            rm_exp = np.concatenate([np.zeros(R - L), np.ones(L)])
            if not np.array_equal(rm[b].astype(int), rm_exp.astype(int)):
                report(f"RESPONSE_MASK {tag}: sum={int(rm[b].sum())}, expected {L} on the last columns")
                ok = False
        if np.abs(adv[b, :R - L]).max(initial=0.0) > 0:
            report(f"ADVANTAGE ON PADDING {tag}")
            ok = False
        a = adv[b, R - L:]
        if a.max() - a.min() > 1e-6:
            report(f"ADVANTAGE NOT CONSTANT OVER RESPONSE {tag}: min={a.min():.4g} max={a.max():.4g} "
                   f"(expected one GRPO scalar per sequence)")
            ok = False
        if ok:
            n_ok += 1

    print(f"\nmatched+ok {n_ok}   problems {len(problems)}   unmatched {n_unmatched}   of {B}")
    for k in ("response_mask", "rollout_logprobs", "base_action_log_probs", "action_log_probs"):
        if k in found:
            print(f"{k:24s} shape {found[k].shape}  (loss_mask shape {lm.shape})")
    if "rollout_logprobs" in found and found["rollout_logprobs"].shape == lm.shape:
        print(f"rollout_logprobs all zero: {bool((found['rollout_logprobs'] == 0).all())} "
              "(expected True unless a ref_logprobs column was supplied)")
    if "rewards" in found:
        print(f"rewards in dump: {np.round(found['rewards'].reshape(B, -1)[:, -1], 3).tolist()[:16]}")
    # vLLM (rollout_logprobs) vs trainer (action_log_probs) on identical weights: where does the error live?
    if "rollout_logprobs" in found and "action_log_probs" in found and found["action_log_probs"].shape == lm.shape:
        rl, al = found["rollout_logprobs"].astype(np.float64), found["action_log_probs"].astype(np.float64)
        m = lm > 0
        d = np.abs(al - rl)
        # position of each column within its own response (responses are right-aligned)
        resp_len = (rm.sum(1) if rm is not None else m.sum(1)).astype(int)
        col = np.arange(R)[None, :]
        pos = col - (R - resp_len)[:, None]          # 0 = first response token
        print(f"\nvLLM vs trainer log-probs on loss tokens: mean |diff| {d[m].mean():.4f}, "
              f"median {np.median(d[m]):.4f}, p90 {np.percentile(d[m], 90):.3f}, max {d[m].max():.2f}, "
              f"frac |diff|>0.2 (outside the 0.8-1.2 band) {(d[m] > 0.2).mean():.3f}, "
              f"mean signed (trainer - vLLM) {(al - rl)[m].mean():+.4f}")
        print("  by position in response:")
        for lo, hi in [(0, 256), (256, 1024), (1024, 4096), (4096, 8192), (8192, 16384), (16384, 10**9)]:
            sel = m & (pos >= lo) & (pos < hi)
            if sel.sum():
                print(f"    {lo:>6}-{hi if hi < 10**9 else 'end':<6} n={sel.sum():>8}  mean |diff| {d[sel].mean():.4f}  "
                      f"p90 {np.percentile(d[sel], 90):.3f}  frac>0.2 {(d[sel] > 0.2).mean():.3f}")
        print("  by vLLM log-prob of the token (confident tokens should agree best):")
        for lo, hi in [(-1e9, -8), (-8, -4), (-4, -2), (-2, -1), (-1, -0.3), (-0.3, -0.05), (-0.05, 0.01)]:
            sel = m & (rl >= lo) & (rl < hi)
            if sel.sum():
                print(f"    [{lo:>6.1f},{hi:>6.2f}) n={sel.sum():>8}  mean |diff| {d[sel].mean():.4f}  frac>0.2 {(d[sel] > 0.2).mean():.3f}")
        # per-sequence summary for the worst sequences
        per_seq = [(d[b][m[b]].mean() if m[b].any() else 0.0, b) for b in range(B)]
        worst = sorted(per_seq, reverse=True)[:5]
        print("  worst sequences (mean |diff|, seq index, response length):",
              [(round(v, 3), b, int(resp_len[b])) for v, b in worst])
    elif "rollout_logprobs" in found:
        print("\n(no action_log_probs in the dump: run one step with policy_loss_type=regular to compare "
              "vLLM vs trainer log-probs)")

    print("\nNote: advantages are non-zero on tool-output tokens by design; the loss_mask is applied in "
          "the loss (reduce_loss / compute_approx_kl / entropy masked_mean). See replay/test_loss_masking.py.")
    sys.exit(0 if (not problems and n_ok > 0) else 1)


if __name__ == "__main__":
    main()