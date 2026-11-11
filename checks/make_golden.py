#!/usr/bin/env python
"""
Build <model dir>/golden_tokenization.jsonl, the ground truth of the tokenizer check, from data
whose token ids are trusted (ids produced by the pipeline that trained / sampled the model).

    python -m checks.make_golden --data <rollouts.jsonl | pregen.parquet> --out <model dir>/golden_tokenization.jsonl [--n 300]

Input: the tokenized rollout JSONL (keys uuid, prompt, prompt_token_ids) or the pregenerated parquet
(prompt as a JSON string, prompt_ids, rep). One line per distinct prompt is written; prompts with
a system message are taken first-come like the others, so both kinds end up in the file.
"""
import argparse
import json


def read_prompts(path, limit):
    out = []
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for r in pq.read_table(path, columns=["prompt", "prompt_ids", "rep"]).to_pylist():
            if r["rep"] == 0:
                out.append((json.loads(r["prompt"]), list(r["prompt_ids"])))
            if len(out) >= limit:
                break
    else:
        seen = set()
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["uuid"] not in seen:
                    seen.add(r["uuid"])
                    out.append((r["prompt"], r["prompt_token_ids"]))
                if len(out) >= limit:
                    break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    prompts = read_prompts(args.data, args.n)
    with open(args.out, "w") as f:
        for messages, ids in prompts:
            f.write(json.dumps({"messages": messages, "ids": ids}, ensure_ascii=False) + "\n")
    n_sys = sum(1 for m, _ in prompts if m and m[0]["role"] == "system")
    print("wrote {} prompts ({} with a system message) to {}".format(len(prompts), n_sys, args.out))


if __name__ == "__main__":
    main()
