"""
Datasets for the judge env: the rows SkyRL's PromptDataset reads (prompt, env_class, reward_spec, extra_info).
Compute nodes are offline: run this on a login node, inside the container.

gsm8k: the parquet files of configs/single_step/gsm8k.yaml with the train rows switched to the judge env.
The validation rows keep env_class "gsm8k": evaluation stays rule-based, independent of the judge.

    python -m skyrl_extensions.judge.dataset gsm8k --src data --out data/gsm8k_judge

science: the training data, a collection of datasets. Each input is a jsonl in the Nemotron-RL format
(responses_create_params.input = the prompt messages, expected_answer, problem; so_openq.jsonl, or the same
records with rollouts: only the prompt and the reference answer are read). One row per question. Every row
carries its dataset's name (data_source: reward is logged per dataset, environment/reward_<name>) and its
judge (reward_spec.judge), so datasets graded by different judges can be mixed in one run.

    python -m skyrl_extensions.judge.dataset science --out data/so_openq_judge \
        --input so_openq.jsonl other=/path/to/another.jsonl [--judge science] [--val-size 200] [--drop-system]

    --input         one or more files; NAME=FILE sets the dataset's name (default: the file name without .jsonl)
    --judge         the judge of these rows (judges.py). Datasets with different judges: one call per judge
                    with --append, which adds to the parquet files in --out
    --val-size      validation rows, taken from the end of EACH input (input order is kept, nothing is shuffled)
    --drop-system   leave out system messages, if the file has any. so_openq.jsonl has none: its prompt is one
                    user message, and the tool is only listed in responses_create_params.tools, which is not
                    read (a single generation cannot run a tool; the offline pipeline put the tool description
                    into a system message itself).
"""
import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

ENV = "judge"


def write(rows, path):
    pq.write_table(pa.Table.from_pylist(rows), path)
    print("{}: {} rows, env_class {}, data_source {}".format(
        path, len(rows), sorted({r["env_class"] for r in rows}), sorted({r["data_source"] for r in rows})))


def gsm8k(args):
    for name, env_class in (("train.parquet", ENV), ("validation.parquet", None)):
        rows = pq.read_table(os.path.join(args.src, name)).to_pylist()
        for r in rows:
            assert r["reward_spec"]["ground_truth"] is not None and r["extra_info"]["answer"], "not a gsm8k_dataset.py file"
            r["env_class"] = env_class or r["env_class"]
        write(rows, os.path.join(args.out, name))


def science(args):
    train, val = [], []
    for item in args.input:
        name, _, path = item.rpartition("=")
        name = name or os.path.basename(path).rsplit(".jsonl", 1)[0]
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                prompt = [{"role": m["role"], "content": m["content"]} for m in rec["responses_create_params"]["input"]]
                assert all(isinstance(m["content"], str) for m in prompt), "{} record {}: message content is not text".format(path, i)
                if args.drop_system:
                    prompt = [m for m in prompt if m["role"] != "system"]
                rows.append({
                    "data_source": name,
                    "prompt": prompt,
                    "env_class": ENV,
                    "reward_spec": {"method": "judge", "judge": args.judge, "ground_truth": str(rec["expected_answer"])},
                    "extra_info": {"uuid": str(rec.get("uuid", i)), "problem": str(rec.get("problem", "")), "topic": str((rec.get("metadata") or {}).get("topic", ""))},
                })
        assert 0 <= args.val_size < len(rows), "--val-size {} with {} records in {}".format(args.val_size, len(rows), path)
        cut = len(rows) - args.val_size
        print("{}: {} train + {} validation rows as '{}', judge {}".format(path, cut, args.val_size, name, args.judge))
        train, val = train + rows[:cut], val + rows[cut:]
    for name, rows in (("train.parquet", train), ("validation.parquet", val)):
        path = os.path.join(args.out, name)
        if args.append and os.path.exists(path):
            rows = pq.read_table(path).to_pylist() + rows
        if rows:
            write(rows, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="task", required=True)
    g = sub.add_parser("gsm8k")
    g.add_argument("--src", required=True, help="dir with train.parquet and validation.parquet of gsm8k_dataset.py")
    g.add_argument("--out", required=True)
    s = sub.add_parser("science")
    s.add_argument("--input", required=True, nargs="+", help="[NAME=]FILE.jsonl, one per dataset")
    s.add_argument("--judge", default="science", help="judge of these rows (skyrl_extensions/judge/judges.py)")
    s.add_argument("--append", action="store_true", help="add to the parquet files already in --out")
    s.add_argument("--out", required=True)
    s.add_argument("--val-size", type=int, default=200)
    s.add_argument("--drop-system", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"gsm8k": gsm8k, "science": science}[args.task](args)


if __name__ == "__main__":
    main()
