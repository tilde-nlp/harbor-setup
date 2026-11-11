#!/usr/bin/env python
"""
The one check that needs GPUs: do vLLM and transformers assign the same log-probs to the same
tokens of the same model directory?  Run through checks/gpu_parity.sbatch (one node, ~20 min).

    python -m checks.gpu_parity vllm    --model DIR --out a.npz [--tp 4]
    python -m checks.gpu_parity hf      --model DIR --out b.npz
    python -m checks.gpu_parity compare a.npz b.npz --model DIR --root REPO [--job ID]

Both scorers read the model's golden file (checks/golden/<model dir name>.jsonl) and store log p(token | previous tokens)
for every token but the first of each prompt, as token ids in, no tokenizer involved, temperature 1.
`compare` prints the difference and writes the verdict to <root>/.check_cache/, where
`./check.sh pre` picks it up (settings.gpu_parity_record). The record is tied to the content of the
model dir's small files: change the model json or the tokenizer files and it has to be rerun.

Plain transformers, no FSDP, no Ray, no sequence parallelism: what differs here differs because
the two libraries build different networks from the same files (notes/ROPE_SCALE_MISMATCH.md).
The trainer stack on top of transformers is covered at step 1 of a run (./check.sh run).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Measured on TildeOpen, 200 stored rollouts (jobs 23611214, 23641346):
#   with the RoPE bug    mean |diff| 0.243, mean signed diff +0.107 nats/token
#   with equal RoPE      mean |diff| 0.027, mean signed diff +0.001   (median 0.004: bf16 noise with a long tail)
MAX_MEAN_ABS_DIFF = 0.05
MAX_MEAN_SIGNED_DIFF = 0.01   # a systematic offset is the signature of two different networks


def golden_ids(model_dir, max_len):
    from checks.tokenizer import golden_path, read_golden

    rows = [g["ids"] for g in read_golden(golden_path(model_dir)) if 2 <= len(g["ids"]) <= max_len]
    print("{} prompts, {} tokens".format(len(rows), sum(len(r) for r in rows)), flush=True)
    return rows


def save(path, logprobs, meta):
    np.savez(path, lens=np.array([len(x) for x in logprobs]), lp=np.concatenate(logprobs).astype(np.float32), meta=json.dumps(meta))
    print("wrote {}: {} tokens, mean log-prob {:.4f}".format(path, sum(len(x) for x in logprobs), float(np.concatenate(logprobs).mean())), flush=True)


def cmd_vllm(args):
    rows = golden_ids(args.model, args.max_len)
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_len + 8, skip_tokenizer_init=True,
              max_num_batched_tokens=8192, gpu_memory_utilization=0.7, max_logprobs=1, dtype="bfloat16")
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)
    outs = llm.generate([{"prompt_token_ids": r} for r in rows], sp, use_tqdm=False)
    lps = [np.array([o.prompt_logprobs[k][r[k]].logprob for k in range(1, len(r))], dtype=np.float32) for r, o in zip(rows, outs)]
    save(args.out, lps, {"engine": "vllm", "model": args.model})


def cmd_hf(args):
    rows = golden_ids(args.model, args.max_len)
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    dev = model.get_input_embeddings().weight.device
    lps = []
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([r], device=dev)
            logits = model(ids).logits[0].float()
            lp = torch.log_softmax(logits[:-1], dim=-1).gather(-1, ids[0, 1:].to(logits.device)[:, None])[:, 0]
            lps.append(lp.cpu().numpy())
    save(args.out, lps, {"engine": "transformers", "model": args.model})


def verdict(a, b):
    """a, b: flat log-prob arrays of the same tokens -> dict with the numbers and PASS/FAIL"""
    d = a - b
    out = {"tokens": int(len(d)), "mean_abs_diff": float(np.abs(d).mean()), "median_abs_diff": float(np.median(np.abs(d))),
           "p99_abs_diff": float(np.percentile(np.abs(d), 99)), "mean_signed_diff": float(d.mean())}
    ok = out["mean_abs_diff"] <= MAX_MEAN_ABS_DIFF and abs(out["mean_signed_diff"]) <= MAX_MEAN_SIGNED_DIFF
    out["status"] = "PASS" if ok else "FAIL"
    return out


def cmd_compare(args):
    from checks.settings import gpu_record_path

    a, b = np.load(args.a), np.load(args.b)
    if not np.array_equal(a["lens"], b["lens"]):
        sys.exit("the two files do not hold the same prompts")
    out = verdict(a["lp"], b["lp"])
    out.update(job=args.job, when=time.strftime("%Y-%m-%d"), model=args.model)
    print("vLLM vs transformers on {tokens} tokens: mean |diff| {mean_abs_diff:.4f} (limit {})  median {median_abs_diff:.4f}  "
          "p99 {p99_abs_diff:.3f}  mean signed {mean_signed_diff:+.4f} (limit {})".format(MAX_MEAN_ABS_DIFF, MAX_MEAN_SIGNED_DIFF, **out))
    print(out["status"] + ("" if out["status"] == "PASS" else
                           ": the two libraries do not compute the same network from this model dir. ./check.sh pre <run yaml> first"))
    path = gpu_record_path(args.root, args.model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("recorded in {}".format(path))
    sys.exit(0 if out["status"] == "PASS" else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("vllm", cmd_vllm), ("hf", cmd_hf)):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--max-len", type=int, default=4096)
        s.add_argument("--tp", type=int, default=4)
        s.set_defaults(fn=fn)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--model", required=True)
    c.add_argument("--root", default=ROOT)
    c.add_argument("--job", default="manual")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
