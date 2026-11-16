#!/usr/bin/env bash
# Show how the tokenizer of a model directory tokenizes a prompt: one row per token (position, id, piece).
# Control tokens are marked, and "▁" in a piece is the leading whitespace: after a control token the first
# text piece must start with it (notes/TRAINING_SANITY_CHECK.md, section 4).
#
#   tools/show_tokens.sh [--model tildeopen] [--system TEXT] [--raw] [TEXT]
#
#   TEXT        the user message (default: a short science question)
#   --system    add a system message
#   --raw       tokenize TEXT as it is, without the chat template
#   --model     model directory, relative to the repo (default: tildeopen)
#
# Runs on a login node (no GPU) with the container's Python, like ./check.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
SANDBOX="${SANDBOX:-$ROOT/multitorch_sandbox.sif}"
BINDS="${CHECK_BINDS:-/scratch/project_465003062,/pfs}"
[ -e "$SANDBOX" ] || { echo "show_tokens.sh: container $SANDBOX missing (a fresh clone needs ./setup.sh first)" >&2; exit 1; }
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
cd "$ROOT"

singularity exec -B "$BINDS" "$SANDBOX" /opt/venv/bin/python - "$@" <<'EOF'
import argparse
import sys

DEFAULT = "Why is the sky blue?\n\nPut your final answer in \\boxed{}."


def main():
    p = argparse.ArgumentParser(prog="tools/show_tokens.sh")
    p.add_argument("text", nargs="?", default=DEFAULT)
    p.add_argument("--model", default="tildeopen")
    p.add_argument("--system")
    p.add_argument("--raw", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    color = sys.stdout.isatty()
    red = (lambda s: "\033[1;31m" + s + "\033[0m") if color else (lambda s: s)
    green = (lambda s: "\033[32m" + s + "\033[0m") if color else (lambda s: s)

    print("tokenizer: {} ({})".format(args.model, type(tok).__name__))
    print("control tokens:")
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        t = getattr(tok, name, None)
        print("  {:10} {!r:20} id {}".format(name, t, tok.convert_tokens_to_ids(t) if t else None))
    named = {getattr(tok, n, None) for n in ("bos_token", "eos_token", "pad_token", "unk_token")}
    for t in tok.all_special_tokens:
        if t not in named:
            print("  {:10} {!r:20} id {}".format("", t, tok.convert_tokens_to_ids(t)))

    if args.raw:
        ids = tok(args.text, add_special_tokens=False).input_ids
    else:
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": args.text}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        ids = list(ids if isinstance(ids, list) else ids["input_ids"])
        print("\nprompt as text:\n" + tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

    special = set(tok.all_special_ids)
    pieces = tok.convert_ids_to_tokens(ids)
    print("\n{:>4}  {:>7}  piece".format("pos", "id"))
    after_control = False
    for i, (tid, piece) in enumerate(zip(ids, pieces)):
        shown = piece.replace("\n", "\\n")
        if tid in special:
            line, note = red(shown), "  <- control"
        else:
            line = shown.replace("▁", green("▁"))
            # the first text piece after a control token: does it carry the leading whitespace?
            note = "" if not after_control else ("  <- leading ▁" if piece.startswith("▁") else "  <- NO leading ▁")
        print("{:>4}  {:>7}  {}{}".format(i, tid, line, note))
        after_control = tid in special

    print("\n{} tokens, {} control".format(len(ids), sum(t in special for t in ids)))
    print("ids: {}".format(ids))


if __name__ == "__main__":
    main()
EOF
