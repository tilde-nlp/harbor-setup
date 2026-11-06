#!/usr/bin/env python
"""
Verify the loss masks of the tokenized rollouts with the real tokenizer.

Reads either the tokenized JSONL (one row per rollout, README keys) or the replay parquet.
For every row it decodes the mask-0 runs and the mask-1 text and checks:

  shape     len(mask) == len(ids), values in {0, 1}, at least one loss token
  prompt    starts with <|system|>=6 or <|user|>=7, ends with <|assistant|>=8,
            6 appears 0 or 2 times, 7 exactly 2 times, 8 exactly once
  control   no 6/7/8/48 inside the rollout except a closing [8, 48]
  closing   if the rollout ends with [8, 48], both have mask 1
  masked    every mask-0 run decodes to text that starts with <|tool_output_start|> and ends
            with <|tool_output_end|>, and contains no second <|tool_output_start|>
  fake      <|tool_output_start|> inside mask-1 text (a fake tool output the model wrote)
            is allowed only when score == 0 (the reward rule)
  roundtrip decode(all ids) == `rollout` text (JSONL input only; whitespace-tolerant; warning)

With --source <so_openq.rollouts.train.jsonl> (one record per prompt, `rollouts[k].tool_calls`
and `n_fake_tool_outputs`), it additionally checks per rollout:
  source    number of mask-0 runs == len(tool_calls), each run decodes to
            <|tool_output_start|> + tool_calls[i].output + <|tool_output_end|> in order,
            and the number of fake markers in mask-1 text == n_fake_tool_outputs

Prints a failure line per row that fails (uuid, rollout_number, check, detail), summary counts,
and --examples N human-readable rows showing masked and trained text side by side.
Exit code 1 if any hard check fails.

Usage:
    python -m replay.verify_masks --jsonl so_openq.rollouts.train.tokens.jsonl --model /path/to/anonymous-17010-slow
    python -m replay.verify_masks --parquet replay.parquet --model ... --source ../so_openq.rollouts.train.jsonl --examples 5
"""
import argparse
import json
import sys
from collections import Counter

TOOL_OUT_START = "<|tool_output_start|>"
TOOL_OUT_END = "<|tool_output_end|>"
SYSTEM_ID, USER_ID, ASSISTANT_ID, EOS_ID = 6, 7, 8, 48
CONTROL = {SYSTEM_ID, USER_ID, ASSISTANT_ID, EOS_ID}


def runs(mask, value):
    """(start, end) index pairs of maximal runs where mask == value."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m == value and start is None:
            start = i
        if m != value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def iter_rows(args):
    if args.jsonl:
        with open(args.jsonl) as f:
            for line in f:
                r = json.loads(line)
                yield {
                    "uid": r["uuid"], "rep": r["rollout_number"], "prompt_ids": r["prompt_token_ids"],
                    "ids": r["rollout_token_ids"], "mask": r["rollout_loss_mask"], "score": r["score"],
                    "text": r.get("rollout"),
                }
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(args.parquet)
        for batch in pf.iter_batches(batch_size=512):
            for r in batch.to_pylist():
                yield {
                    "uid": r["uid"], "rep": r["rep"], "prompt_ids": r["prompt_ids"], "ids": r["response_ids"],
                    "mask": [int(m) for m in r["loss_mask"]], "score": r["score"], "text": None,
                }


def load_source(path):
    """(uuid, k) -> (list of expected tool output strings, n_fake_tool_outputs)."""
    src = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for k, ro in enumerate(rec["rollouts"]):
                outs = [tc["output"] for tc in (ro.get("tool_calls") or [])]
                src[(rec["uuid"], k)] = (outs, ro.get("n_fake_tool_outputs", 0))
    return src


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--jsonl")
    g.add_argument("--parquet")
    ap.add_argument("--model", required=True, help="tokenizer dir of anonymous-17010-slow")
    ap.add_argument("--source", help="original rollouts JSONL (one record per prompt) for the tool_calls cross-check")
    ap.add_argument("--examples", type=int, default=3, help="print this many rows with tool outputs in readable form")
    ap.add_argument("--max-failures", type=int, default=50, help="stop printing failures after this many")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.eos_token_id == EOS_ID, f"eos_token_id is {tok.eos_token_id}, expected 48"
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)

    source = load_source(args.source) if args.source else None
    stats, fails = Counter(), Counter()
    printed_fail, printed_ex = 0, 0

    def fail(row, check, detail):
        nonlocal printed_fail
        fails[check] += 1
        if printed_fail < args.max_failures:
            print(f"FAIL {row['uid']} rep={row['rep']} [{check}] {detail}", file=sys.stderr)
            printed_fail += 1

    for row in iter_rows(args):
        stats["rows"] += 1
        ids, mask, p = row["ids"], row["mask"], row["prompt_ids"]

        # shape
        if len(ids) != len(mask) or not set(mask) <= {0, 1} or sum(mask) == 0:
            fail(row, "shape", f"len ids={len(ids)} mask={len(mask)} values={set(mask)} loss_tokens={sum(mask)}")
            continue
        # prompt
        c = Counter(p)
        if not p or p[0] not in (SYSTEM_ID, USER_ID) or p[-1] != ASSISTANT_ID or c[SYSTEM_ID] not in (0, 2) \
                or c[USER_ID] != 2 or c[ASSISTANT_ID] != 1 or c[EOS_ID] != 0:
            fail(row, "prompt", f"first={p[:1]} last={p[-1:]} counts 6={c[6]} 7={c[7]} 8={c[8]} 48={c[48]}")
        # control tokens inside the rollout
        ends_stop = ids[-2:] == [ASSISTANT_ID, EOS_ID]
        body = ids[:-2] if ends_stop else ids
        inner = [t for t in body if t in CONTROL]
        if inner:
            fail(row, "control", f"control ids inside rollout: {Counter(inner)} ends_stop={ends_stop}")
        if ends_stop:
            stats["rows_stop"] += 1
            if mask[-1] != 1 or mask[-2] != 1:
                fail(row, "closing", f"mask on closing [8,48] is {mask[-2:]}")
        else:
            stats["rows_no_stop"] += 1

        # masked runs
        zero_runs = runs(mask, 0)
        stats["masked_runs"] += len(zero_runs)
        stats["masked_tokens"] += sum(e - s for s, e in zero_runs)
        stats["loss_tokens"] += sum(mask)
        if zero_runs:
            stats["rows_with_masked"] += 1
        decoded_runs = []
        for s, e in zero_runs:
            t = dec(ids[s:e])
            decoded_runs.append(t)
            ts = t.lstrip()
            if not ts.startswith(TOOL_OUT_START) or not ts.rstrip().endswith(TOOL_OUT_END) \
                    or ts.count(TOOL_OUT_START) != 1:
                fail(row, "masked", f"run [{s}:{e}] decodes to {t[:60]!r}...{t[-40:]!r}")
        # fake tool outputs in the loss
        kept_text = dec([t for t, m in zip(ids, mask) if m == 1])
        n_fake = kept_text.count(TOOL_OUT_START)
        if n_fake:
            stats["rows_with_fake_in_loss"] += 1
            if row["score"] != 0:
                fail(row, "fake", f"{n_fake} fake tool output(s) in loss but score={row['score']}")
        # roundtrip (JSONL only)
        if row["text"] is not None:
            full = dec(ids)
            if full.strip() != row["text"].strip():
                stats["warn_roundtrip_mismatch"] += 1
                if stats["warn_roundtrip_mismatch"] <= 3:
                    print(f"WARN {row['uid']} rep={row['rep']} decode(ids) != rollout text "
                          f"(len {len(full)} vs {len(row['text'])})", file=sys.stderr)
        # cross-check with the source rollouts
        if source is not None:
            key = (row["uid"], row["rep"])
            if key not in source:
                fail(row, "source", "row not found in --source")
            else:
                outs, n_fake_src = source[key]
                if len(outs) != len(zero_runs):
                    fail(row, "source", f"{len(zero_runs)} masked runs but {len(outs)} tool_calls in source")
                else:
                    for i, (t, o) in enumerate(zip(decoded_runs, outs)):
                        exp = TOOL_OUT_START + o + TOOL_OUT_END
                        if t.lstrip() != exp and t.strip() != exp.strip():
                            fail(row, "source", f"masked run {i} != tool_calls[{i}].output "
                                                f"(decoded {t.strip()[:50]!r}, expected {exp.strip()[:50]!r})")
                            break
                if n_fake != n_fake_src:
                    fail(row, "source", f"{n_fake} fake markers in loss text, source says n_fake_tool_outputs={n_fake_src}")

        # examples
        if zero_runs and printed_ex < args.examples:
            printed_ex += 1
            print(f"\n=== example {row['uid']} rep={row['rep']} score={row['score']} "
                  f"loss_tokens={sum(mask)}/{len(mask)} masked_runs={len(zero_runs)} ===")
            pos = 0
            for s, e in zero_runs:
                print("  TRAIN:", repr(dec(ids[pos:s])[-200:]))
                print("  MASK :", repr(dec(ids[s:e])[:300]))
                pos = e
            print("  TRAIN:", repr(dec(ids[pos:])[:200]))

        if args.limit and stats["rows"] >= args.limit:
            break

    print("\n--- summary ---")
    for k in sorted(stats):
        print(f"{k:28s} {stats[k]}")
    if fails:
        print("--- failures ---")
        for k in sorted(fails):
            print(f"{k:28s} {fails[k]}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()