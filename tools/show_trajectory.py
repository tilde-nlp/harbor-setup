#!/usr/bin/env python3
"""
Write a Harbor agent trajectory (ATIF trajectory.json, e.g. terminus-2) as readable, coloured text into a .txt
file next to it: test.json -> test.txt, trials/<trial>/agent/trajectory.json -> .../trajectory.txt.

    tools/show_trajectory.py test.json                 a trajectory file
    tools/show_trajectory.py trials/<trial>            a trial dir (reads agent/trajectory.json)
    tools/show_trajectory.py test.json --steps 2-4     only some steps (1-based, as step_id)
    tools/show_trajectory.py test.json --no-prompt     skip the long first user message (the task prompt)
    tools/show_trajectory.py test.json --no-obs        skip the terminal output

    less -R test.txt                                   read it with the colours

Each step shows: who spoke (user / agent), when, its tokens, what the model actually generated (its
completion_token_ids decoded with the tokenizer, thinking included; the "message" field only keeps the parsed
Analysis/Plan), the tool calls (the keystrokes sent to the terminal, verbatim) and the observation (terminal
output sent back, without the blank rows the screen is padded with).

Decoding needs `tokenizers` or `transformers` and the tokenizer dir (--model, default: tildeopen in the repo).
If the Python running this has neither library, the script runs itself again inside the container
(multitorch_sandbox.sif, override with $SANDBOX), like tools/show_tokens.sh.
"""
import argparse
import datetime
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BOLD, DIM, CYAN, GREEN, YELLOW, MAGENTA = "1", "2", "36", "32", "33", "35"
WHO = {"user": YELLOW, "agent": CYAN, "system": MAGENTA}


def c(code, s):
    return "\033[%sm%s\033[0m" % (code, s)


def find_trajectory(path):
    if os.path.isdir(path):
        for cand in ("agent/trajectory.json", "trajectory.json"):
            if os.path.exists(os.path.join(path, cand)):
                return os.path.join(path, cand)
        sys.exit("show_trajectory.py: no agent/trajectory.json under %s" % path)
    if not os.path.exists(path):
        sys.exit("show_trajectory.py: %s not found" % path)
    return path


def parse_range(spec):
    """'2-4,7' -> {2, 3, 4, 7}"""
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def ts(s):
    try:
        return datetime.datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def strip_screen(text):
    """Drop the empty rows tmux pads the terminal screen with."""
    return "\n".join(line.rstrip() for line in text.rstrip().split("\n"))


def text_of(content):
    if isinstance(content, list):  # multimodal content parts
        return "\n".join(part.get("text", "<%s>" % part.get("type")) for part in content)
    return content or ""


def rerun_in_container():
    """No tokenizer library here: run this same command with the container's Python."""
    sandbox = os.environ.get("SANDBOX", os.path.join(ROOT, "multitorch_sandbox.sif"))
    if os.environ.get("SHOW_TRAJECTORY_IN_CONTAINER") or not os.path.exists(sandbox) or not shutil.which("singularity"):
        sys.exit("show_trajectory.py: decoding needs `tokenizers` or `transformers`, and there is no container "
                 "(%s) to run in" % sandbox)
    binds = os.environ.get("CHECK_BINDS", "/scratch/project_465003062,/pfs")
    os.environ.update(SHOW_TRAJECTORY_IN_CONTAINER="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", TRANSFORMERS_VERBOSITY="error")
    os.execvp("singularity", ["singularity", "exec", "-B", binds, "-B", os.getcwd(), sandbox,
                              "/opt/venv/bin/python", os.path.realpath(__file__)] + sys.argv[1:])


def make_decoder(model):
    if not os.path.isdir(model):
        sys.exit("show_trajectory.py: tokenizer dir %s not found (--model)" % model)
    try:
        from tokenizers import Tokenizer
        if os.path.exists(os.path.join(model, "tokenizer.json")):
            tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
            return lambda ids: tok.decode(ids, skip_special_tokens=False)
    except ImportError:
        pass
    try:
        from transformers import AutoTokenizer
    except ImportError:
        rerun_in_container()
    tok = AutoTokenizer.from_pretrained(model)
    return lambda ids: tok.decode(ids, skip_special_tokens=False)


def render(traj, decode, out, only=None, no_prompt=False, no_obs=False):
    def w(s=""):
        out.write(s + "\n")

    agent = traj.get("agent", {})
    steps = traj.get("steps", [])
    fm = traj.get("final_metrics", {})
    w(c(BOLD, "Trajectory %s" % traj.get("session_id", "?")))
    w("agent  %s %s   model %s" % (agent.get("name", "?"), agent.get("version", ""), agent.get("model_name", "?")))
    t0 = ts(steps[0].get("timestamp")) if steps else None
    t1 = ts(steps[-1].get("timestamp")) if steps else None
    took = " in %.0fs" % (t1 - t0).total_seconds() if t0 and t1 else ""
    w("steps  %d%s   tokens: prompt %s, completion %s, cached %s" % (
        len(steps), took, fm.get("total_prompt_tokens", "?"), fm.get("total_completion_tokens", "?"),
        fm.get("total_cached_tokens", "?")))

    for i, step in enumerate(steps):
        sid = step.get("step_id", i + 1)
        if only is not None and sid not in only:
            continue
        source = step.get("source", "?")
        t = ts(step.get("timestamp"))
        head = "step %s · %s" % (sid, source)
        if t and t0:
            head += " · +%.0fs" % (t - t0).total_seconds()
        m = step.get("metrics") or {}
        if "prompt_tokens" in m:
            head += " · prompt %s, completion %s tok" % (m.get("prompt_tokens"), m.get("completion_tokens"))
        w()
        w(c(BOLD + ";" + WHO.get(source, "0"), "━━ " + head + " " + "━" * max(3, 70 - len(head))))

        if i == 0 and source == "user" and no_prompt:
            w(c(DIM, "(task prompt, %d chars, skipped)" % len(text_of(step.get("message")))))
            continue

        if m.get("completion_token_ids"):
            w(decode(m["completion_token_ids"]).strip())
        else:  # user / system steps, or an agent step without token ids
            if step.get("reasoning_content"):
                w(c(DIM, step["reasoning_content"].strip()))
            msg = text_of(step.get("message")).strip()
            if msg:
                w(msg)

        for call in step.get("tool_calls") or []:
            name = call.get("function_name", "?")
            a = call.get("arguments") or {}
            if "keystrokes" in a:
                wait = a.get("duration")
                w(c(GREEN, "── %s%s" % (name, " (wait %ss)" % wait if wait is not None else "")))
                w(c(GREEN, a["keystrokes"].rstrip("\n")))
            else:
                w(c(GREEN, "── " + name + (" " + json.dumps(a) if a else "")))

        if not no_obs:
            for r in (step.get("observation") or {}).get("results", []):
                content = text_of(r.get("content"))
                if content:
                    w(c(DIM, "── observation"))
                    w(strip_screen(content))


def main():
    p = argparse.ArgumentParser(prog="tools/show_trajectory.py", description=__doc__.split("\n\n")[0].strip())
    p.add_argument("path", help="trajectory.json or a trial dir")
    p.add_argument("--steps", help="only these step ids, e.g. 2-4,7")
    p.add_argument("--no-prompt", action="store_true", help="skip the first user message")
    p.add_argument("--no-obs", action="store_true", help="skip observations (terminal output)")
    p.add_argument("--model", default=os.path.join(ROOT, "tildeopen"),
                   help="tokenizer dir (default: tildeopen in the repo)")
    args = p.parse_args()

    path = find_trajectory(args.path)
    with open(path) as f:
        traj = json.load(f)
    decode = make_decoder(args.model)
    out_path = os.path.splitext(path)[0] + ".txt"
    with open(out_path, "w") as out:
        render(traj, decode, out, parse_range(args.steps) if args.steps else None, args.no_prompt, args.no_obs)
    print("wrote %s   (read with: less -R %s)" % (out_path, out_path))


if __name__ == "__main__":
    main()
