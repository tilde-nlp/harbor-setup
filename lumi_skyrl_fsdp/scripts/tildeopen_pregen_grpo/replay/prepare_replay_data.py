#!/usr/bin/env python
"""
Convert the pre-tokenized train split (`so_openq.rollouts.train.tokens.jsonl`, one row per
rollout, see its README.md) into the replay parquet that replay/generator.py reads.

Input row keys used:
    uuid, rollout_number, prompt, prompt_token_ids, rollout_token_ids, rollout_loss_mask,
    score, answer_true
Output parquet, one row per rollout:
    uid           = uuid (GRPO group id)
    rep           = rollout_number (0..n-1)
    prompt        = JSON string of the `prompt` messages (what the trainer logs)
    prompt_ids    = prompt_token_ids
    response_ids  = rollout_token_ids (possibly truncated to --max-seq-len)
    loss_mask     = rollout_loss_mask
    score         = score
    stop_reason   = "stop" if the rollout ends with the closing [8, 48], else "length"
                    (SkyRL's overlong filtering keys on "stop")
    answer_true   = answer_true

Checks: 8 rows per uuid with rollout_number 0..7, contiguous; mask length == ids length;
mask values in {0,1}; prompt starts with 6 or 7 and ends with 8; trailing [8, 48] has mask 1.
Group filter (on by default): keep a prompt only if some rollout has answer_true and the 8
scores are not all equal (all-equal groups give GRPO zero advantage).

Usage:
    python -m replay.prepare_replay_data \
        --jsonl so_openq.rollouts.train.tokens.jsonl --out replay.parquet
"""
import argparse
import json
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM_ID, USER_ID, ASSISTANT_ID, EOS_ID = 6, 7, 8, 48

SCHEMA = pa.schema(
    [
        ("uid", pa.string()),
        ("rep", pa.int32()),
        ("prompt", pa.string()),
        ("prompt_ids", pa.list_(pa.int32())),
        ("response_ids", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.int8())),
        ("score", pa.float32()),
        ("stop_reason", pa.string()),
        ("answer_true", pa.bool_()),
    ]
)


def check_row(row):
    p, r, m = row["prompt_token_ids"], row["rollout_token_ids"], row["rollout_loss_mask"]
    if len(r) != len(m):
        raise ValueError(f"rollout_loss_mask length {len(m)} != rollout_token_ids length {len(r)}")
    if not set(m) <= {0, 1}:
        raise ValueError(f"rollout_loss_mask has values outside 0/1: {set(m) - {0, 1}}")
    if not p or p[0] not in (SYSTEM_ID, USER_ID) or p[-1] != ASSISTANT_ID:
        raise ValueError(f"prompt_token_ids must start with 6/7 and end with 8, got {p[:1]}...{p[-1:]}")
    if len(r) == 0:
        raise ValueError("empty rollout_token_ids")
    ends_stop = r[-2:] == [ASSISTANT_ID, EOS_ID]
    if ends_stop and (m[-1] != 1 or m[-2] != 1):
        raise ValueError("closing [8, 48] must have loss mask 1")
    if sum(m) == 0:
        raise ValueError("rollout has no loss tokens")
    return ends_stop


def keep_group(rows):
    return any(r["answer_true"] for r in rows) and len({r["score"] for r in rows}) > 1


def read_groups(path, n_samples):
    """Yield (uuid, [rows]) for each contiguous run of rows with the same uuid."""
    cur_uid, cur = None, []
    seen = set()
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            row = json.loads(line)
            uid = row["uuid"]
            if uid != cur_uid:
                if cur:
                    yield cur_uid, cur
                if uid in seen:
                    raise ValueError(f"line {ln}: uuid {uid} appears in two separate runs; rows of a prompt must be contiguous")
                seen.add(uid)
                cur_uid, cur = uid, []
            cur.append(row)
    if cur:
        yield cur_uid, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=65536, help="prompt + response cap; longer responses are cut and marked 'length'")
    ap.add_argument("--no-filter", action="store_true", help="keep every group, even with no correct answer or all-equal scores")
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept prompts (smoke tests)")
    args = ap.parse_args()

    stats = Counter()
    max_total = 0
    writer = pq.ParquetWriter(args.out, SCHEMA, compression="zstd")
    buf = {name: [] for name in SCHEMA.names}

    def flush():
        if buf["uid"]:
            writer.write_table(pa.table(buf, schema=SCHEMA))
            for v in buf.values():
                v.clear()

    for uid, rows in read_groups(args.jsonl, args.n_samples):
        stats["prompts_in"] += 1
        reps = sorted(r["rollout_number"] for r in rows)
        if reps != list(range(args.n_samples)):
            stats["prompts_dropped_bad_group"] += 1
            print(f"drop {uid}: rollout_numbers {reps}, expected 0..{args.n_samples - 1}", file=sys.stderr)
            continue
        if not args.no_filter and not keep_group(rows):
            stats["prompts_filtered"] += 1
            continue
        try:
            out_rows = []
            for r in sorted(rows, key=lambda x: x["rollout_number"]):
                ends_stop = check_row(r)
                ids, mask = r["rollout_token_ids"], r["rollout_loss_mask"]
                stop_reason = "stop" if ends_stop else "length"
                room = args.max_seq_len - len(r["prompt_token_ids"])
                if room <= 0:
                    raise ValueError("prompt alone exceeds max_seq_len")
                if len(ids) > room:
                    ids, mask, stop_reason = ids[:room], mask[:room], "length"
                    stats["rollouts_truncated"] += 1
                out_rows.append((r, ids, mask, stop_reason))
        except ValueError as e:
            stats["prompts_dropped_check_failed"] += 1
            print(f"drop {uid}: {e}", file=sys.stderr)
            continue

        prompt_json = json.dumps(rows[0]["prompt"], ensure_ascii=False)
        for r, ids, mask, stop_reason in out_rows:
            buf["uid"].append(uid)
            buf["rep"].append(r["rollout_number"])
            buf["prompt"].append(prompt_json)
            buf["prompt_ids"].append(r["prompt_token_ids"])
            buf["response_ids"].append(ids)
            buf["loss_mask"].append(mask)
            buf["score"].append(float(r["score"]))
            buf["stop_reason"].append(stop_reason)
            buf["answer_true"].append(bool(r["answer_true"]))
            stats["rollouts_out"] += 1
            stats["tokens_prompt"] += len(r["prompt_token_ids"])
            stats["tokens_response"] += len(ids)
            stats["tokens_loss"] += sum(mask)
            stats[f"stop_{stop_reason}"] += 1
            max_total = max(max_total, len(r["prompt_token_ids"]) + len(ids))
        stats["prompts_kept"] += 1
        if len(buf["uid"]) >= 2000:
            flush()
        if args.limit and stats["prompts_kept"] >= args.limit:
            break
    flush()
    writer.close()

    for k in sorted(stats):
        print(f"{k:32s} {stats[k]}")
    if stats["rollouts_out"]:
        n = stats["rollouts_out"]
        print(f"{'avg_prompt_tokens':32s} {stats['tokens_prompt'] / n:.0f}")
        print(f"{'avg_response_tokens':32s} {stats['tokens_response'] / n:.0f}")
        print(f"{'max_prompt_plus_response':32s} {max_total}   <- trainer.algorithm.max_seq_len must be >= this")


if __name__ == "__main__":
    main()