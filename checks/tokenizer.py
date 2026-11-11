"""
Does the tokenizer the trainer and vLLM load from the model directory produce the token ids the
model was trained with?

Ground truth is a golden file: one {"messages": [...], "ids": [...]} per line, the chat-templated
prompt (with generation prompt) as the model must see it. It depends on the tokenizer, not on the
weights, so there is one per model family in the repo: checks/golden/<model dir name>.jsonl.
A <model dir>/golden_tokenization.jsonl, if present, is used instead. Build one with
checks/make_golden.py from data whose token ids are trusted (checks/golden/README.md).

Background: in this container transformers 5.8 loaded TildeOpen's tokenizer.model without the
leading "▁" after control tokens (add_prefix_space=false): 0 of 300 prompts were tokenized the way
the model was trained. notes/TRAINING_SANITY_CHECK.md, section 4.
"""
import collections
import json
import os

from checks.lib import FAIL, PASS, WARN, Result, check, small_model_files

GOLDEN = "golden_tokenization.jsonl"                                             # inside a model dir: override
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")  # in the repo: <model dir name>.jsonl


def golden_path(model_dir):
    """The golden file for a model dir: its own override if it has one, else the repo's file for its name."""
    own = os.path.join(model_dir, GOLDEN)
    if os.path.exists(own):
        return own
    return os.path.join(GOLDEN_DIR, os.path.basename(os.path.realpath(model_dir).rstrip("/")) + ".jsonl")


def read_golden(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def compare(tokenizer_dir, golden):
    """-> (number identical, Counter of where the first difference is, one example or None)"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    same, where, example = 0, collections.Counter(), None
    for g in golden:
        got = tok.apply_chat_template(g["messages"], add_generation_prompt=True, tokenize=True)
        got = list(got if isinstance(got, list) else got["input_ids"])
        ref = g["ids"]
        if got == ref:
            same += 1
            continue
        k = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
        prev = tok.convert_ids_to_tokens(ref[k - 1]) if k else "<start>"
        where["right after {}".format(prev) if prev.startswith("<") else "inside text"] += 1
        if example is None:
            example = "position {}: expected {} got {}".format(
                k, tok.convert_ids_to_tokens(ref[max(0, k - 1): k + 4]), tok.convert_ids_to_tokens(got[max(0, k - 1): k + 4]))
    return same, where, example


def golden_result(tokenizer_dir, golden_file, what):
    if not os.path.exists(golden_file):
        return Result(WARN, "no golden file for this model (checks/golden/{}): tokenization is unchecked".format(os.path.basename(golden_file)),
                      ["a model dir is matched to its golden file by its directory name",
                       "build one from trusted token ids: python -m checks.make_golden --data <rollouts.jsonl|parquet> --out {}".format(golden_file)])
    golden = read_golden(golden_file)
    same, where, example = compare(tokenizer_dir, golden)
    data = {"identical": same, "total": len(golden)}
    if same == len(golden):
        return Result(PASS, "{} reproduces all {} golden prompts".format(what, len(golden)), data=data)
    return Result(FAIL, "{} reproduces {}/{} golden prompts".format(what, same, len(golden)),
                  ["first difference: {}".format(dict(where)), example,
                   "the model is prompted with token sequences it was not trained on"], data)


@check("pre", "tokenizer.golden", cache_inputs=lambda ctx: small_model_files(ctx.model_dir) + [golden_path(ctx.model_dir)])
def tokenizer_golden(ctx):
    """The model dir's tokenizer + chat template must reproduce the trusted token ids of the model's golden file."""
    return golden_result(ctx.model_dir, golden_path(ctx.model_dir), "the model dir's tokenizer")
