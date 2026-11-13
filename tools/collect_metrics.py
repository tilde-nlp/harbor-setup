#!/usr/bin/env python3
"""
Collect per-step training metrics from the Slurm .out files of runs into CSV.

    tools/collect_metrics.py                      latest run under previous_runs/
    tools/collect_metrics.py 23683421             a job id, a run dir name (or a part of it) ...
    tools/collect_metrics.py path/to/slurm-1.out  ... a .out file, or a run dir
    tools/collect_metrics.py --all -o all.csv     every run, one table
    tools/collect_metrics.py --list               what is there

Writes two files (default: <run dir>/metrics.csv for one run, ./metrics.csv for several):
    metrics.csv        one row per training step. Columns: run, job_id, step, then every metric SkyRL logs
                       (reward/*, loss/*, policy/*, generate/*, timing/*, trainer/*, ray/*, eval/* ...),
                       time/* (wall clock of the step, from the log timestamps) and log/* (values that are
                       only in log lines, e.g. log/avg_response_length, log/example_total_reward).
    metrics.runs.csv   one row per run: job id, nodes, layout, step count, wall time, and the resolved
                       config.yaml flattened to cfg/<key.path> columns. Join on `run`.

Where the numbers come from. SkyRL's console logger prints every metric rounded to 4 decimals (the learning
rate 3e-6 shows as 0.0000). The same values are also logged at full precision in ordinary log lines of the
step; those are used wherever they exist, the rounded value otherwise. --rounded keeps the console values.

Python 3.6+, standard library only (PyYAML is used for config.yaml when it is installed).
"""
import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RANK = re.compile(r"^\d+: ")                                  # srun --label
RAY = re.compile(r"^\((\w+)(?: pid=\d+)?(?:, ip=[\d.]+)?\) ?")  # ray worker prefix
LOGURU = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \| (\w+) *\| ([\w.]+):([\w<>]+):(\d+) - (.*)$")
STEP_HEADER = re.compile(r"^Step (\d+): ?$")
DICT_ITEM = re.compile(r"^\s*\{?'([^']+)': ('[^']*'|[^,}]+)[,}]\s*$")
NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+|nan|inf)"
KV_LINE = re.compile(r"^[\w/.@-]+: %s(?:, [\w/.@-]+: %s)*$" % (NUM, NUM))
PHASE = re.compile(r"^(Started|Finished): '([^']+)'(?:, time cost: ([\d.]+)s)?$")
EXAMPLE_REWARD = re.compile(r"Output \(Total Reward: (%s)\)" % NUM)
MAX_BLOCK_LINES = 2000  # a metrics block that does not close within this many lines is dropped


def number(text):
    """'0.4121' -> 0.4121, '512' -> 512, anything else -> the text itself."""
    text = text.strip().strip("'\"")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_time(stamp):
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f")


def merge_precise(row, precise):
    """Overwrite the rounded console values in `row` with full-precision ones logged under a shorter name.

    'policy_kl' -> 'policy/policy_kl', 'avg_final_rewards' -> 'loss/avg_final_rewards'; a name no metric
    ends with goes to 'log/<name>'.
    """
    for key, value in precise.items():
        if key in row or key.startswith(("time/", "log/")):
            target = key
        else:
            ends = [k for k in row if k.endswith("/" + key)]
            preferred = [k for k in ends if k == "policy/" + key]
            target = (preferred or ends or ["log/" + key])[0]
        row[target] = value


def parse_out(path, rounded=False):
    """-> (rows, info): one dict per logged step, and what the file says about the run."""
    rows = OrderedDict()   # step -> row; a step logged twice (resumed job) keeps the later one
    info = OrderedDict()
    pending = OrderedDict()  # full-precision values seen since the last metrics block
    block = None             # (step, row, lines read) while inside a "Step N:" metrics block
    step_start = step_end = first_start = None
    hung = False

    with open(path, errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw.rstrip("\n"))
            if line.startswith("run dir: "):
                info["run_dir_logged"] = line[9:].strip()
            elif line.startswith("layout: "):
                info["layout"] = line[8:].strip()
            elif line.startswith("nodes: "):
                info["nodes"] = line[7:].split("head:")[0].strip()
            elif "WATCHDOG:" in line:
                hung = True
            line = RAY.sub("", RANK.sub("", line))

            if block is not None:
                step, row, n = block
                m = DICT_ITEM.match(line)
                if m:
                    row[m.group(1)] = number(m.group(2))
                    if line.rstrip().endswith("}"):
                        if not rounded:
                            merge_precise(row, pending)
                        else:
                            merge_precise(row, {k: v for k, v in pending.items() if k.startswith("time/")})
                        row.setdefault("trainer/global_step", step)
                        rows[step] = row
                        pending = OrderedDict()
                        block = None
                        continue
                block = (step, row, n + 1)
                if n > MAX_BLOCK_LINES:
                    block = None
                continue

            m = EXAMPLE_REWARD.search(line)
            if m:
                pending["log/example_total_reward"] = number(m.group(1))
                continue
            m = LOGURU.search(line)
            if not m:
                continue
            stamp, _level, _module, _func, _lineno, msg = m.groups()
            msg = msg.strip()

            m = STEP_HEADER.match(msg)
            if m:
                block = (int(m.group(1)), OrderedDict(), 0)
                continue
            m = PHASE.match(msg)
            if m:
                if m.group(2) == "step":
                    t = parse_time(stamp)
                    if m.group(1) == "Started":
                        step_start = t
                        first_start = first_start or t
                        pending["time/step_start"] = stamp
                    else:
                        step_end = t
                        pending["time/step_end"] = stamp
                        pending["time/elapsed_s"] = round((t - first_start).total_seconds(), 3) if first_start else ""
                continue
            if msg.startswith("{'") and msg.endswith("}"):
                try:
                    values = ast.literal_eval(msg)
                except (ValueError, SyntaxError):
                    continue
                pending.update((k, v) for k, v in values.items() if isinstance(v, (int, float)))
                continue
            if KV_LINE.match(msg):
                for item in msg.split(", "):
                    key, value = item.rsplit(": ", 1)
                    pending[key] = number(value)

    info["steps_logged"] = len(rows)
    if rows:
        steps = list(rows.values())
        info["first_step"] = next(iter(rows))
        info["last_step"] = next(reversed(rows))
        info["first_step_start"] = steps[0].get("time/step_start", "")
        info["last_step_end"] = steps[-1].get("time/step_end", "")
        info["train_wall_s"] = steps[-1].get("time/elapsed_s", "")
    info["watchdog_hung"] = hung
    return list(rows.values()), info


# ---------------------------------------------------------------------------------------------- run lookup

def job_id_of(out_path):
    m = re.search(r"slurm-(\d+)", os.path.basename(out_path))
    return m.group(1) if m else ""


def outs_in(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "slurm-*.out")), key=lambda p: (len(job_id_of(p)), job_id_of(p)))


def all_runs(runs_dir):
    """Run dirs that have a Slurm log, oldest first (their names start with the submit timestamp)."""
    dirs = sorted(d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d) and not os.path.islink(d))
    return [d for d in dirs if outs_in(d)]


def resolve(spec, runs_dir):
    """A .out file, a run dir, a job id or (part of) a run dir name -> list of .out files."""
    if os.path.isfile(spec):
        return [spec]
    if os.path.isdir(spec):
        outs = outs_in(spec)
        if not outs:
            sys.exit("collect_metrics: no slurm-*.out in %s" % spec)
        return outs
    runs = all_runs(runs_dir)
    hits = [o for d in runs for o in outs_in(d) if job_id_of(o) == spec]
    if not hits:
        hits = [o for d in runs if spec in os.path.basename(d) for o in outs_in(d)]
    if not hits:
        sys.exit("collect_metrics: %r is not a file, a run dir, a job id or a run name under %s" % (spec, runs_dir))
    return hits


# -------------------------------------------------------------------------------------------- run metadata

def tiny_yaml(text):
    """Enough YAML for the rendered config.yaml (nested mappings, lists of scalars) when PyYAML is missing."""
    def scalar(s):
        s = s.strip()
        if s in ("null", "~", ""):
            return None
        if s in ("true", "false"):
            return s == "true"
        if s[:1] in "'\"" and s[-1:] == s[:1]:
            return s[1:-1]
        return number(s)

    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]

    def block(i, indent):
        if i < len(lines) and lines[i].lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- ") and len(lines[i]) - len(lines[i].lstrip()) >= indent:
                items.append(scalar(lines[i].lstrip()[2:]))
                i += 1
            return items, i
        out = OrderedDict()
        while i < len(lines):
            cur = len(lines[i]) - len(lines[i].lstrip())
            if cur < indent or ":" not in lines[i]:
                break
            key, _, rest = lines[i].strip().partition(":")
            i += 1
            if rest.strip():
                out[key] = scalar(rest)
            elif i < len(lines):
                nxt = len(lines[i]) - len(lines[i].lstrip())
                is_list = lines[i].lstrip().startswith("- ")
                if nxt > cur or (is_list and nxt == cur):
                    out[key], i = block(i, nxt)
                else:
                    out[key] = None
            else:
                out[key] = None
        return out, i

    return block(0, 0)[0]


def flatten(value, prefix, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(v, "%s.%s" % (prefix, k) if prefix else str(k), out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value)
    else:
        out[prefix] = "" if value is None else value
    return out


def read_config(run_dir):
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        text = f.read()
    try:
        import yaml
        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = tiny_yaml(text)
    except Exception as e:  # a config that does not parse must not lose the metrics
        sys.stderr.write("collect_metrics: %s: %s\n" % (path, e))
        return {}
    return flatten(cfg or {}, "", OrderedDict())


def read_slurm_job(run_dir):
    """The fields worth having from slurm_job.txt (scontrol show job, taken at job start)."""
    path = os.path.join(run_dir, "slurm_job.txt")
    if not os.path.isfile(path):
        return {}
    with open(path, errors="replace") as f:
        text = f.read()
    out = OrderedDict()
    for field in ("JobName", "Partition", "NumNodes", "NodeList", "SubmitTime", "StartTime", "TimeLimit"):
        m = re.search(r"\b%s=(\S+)" % field, text)
        if m:
            out["slurm/" + field] = m.group(1)
    return out


def read_git(run_dir):
    path = os.path.join(run_dir, "git.txt")
    if not os.path.isfile(path):
        return {}
    out = OrderedDict()
    with open(path, errors="replace") as f:
        for line in f:
            for key in ("commit", "command"):
                if line.startswith(key + ": "):
                    out[key] = line[len(key) + 2:].strip()
    return out


# ------------------------------------------------------------------------------------------------- output

LEAD = ["run", "job_id", "step"]
GROUPS = ["trainer/", "time/", "reward/", "loss/", "policy/", "critic/", "eval/", "generate/", "log/", "timing/", "ray/"]


def column_order(rows):
    seen = OrderedDict()
    for row in rows:
        for key in row:
            seen[key] = True

    def rank(key):
        if key in LEAD:
            return (0, LEAD.index(key), "")
        for i, g in enumerate(GROUPS):
            if key.startswith(g):
                return (1, i, "")
        return (2, 0, "")

    keys = list(seen)
    return sorted(keys, key=lambda k: rank(k) + (keys.index(k),))


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                 epilog="See the top of this file for the columns.")
    ap.add_argument("runs", nargs="*", help=".out file, run dir, job id or (part of) a run dir name; default: latest run")
    ap.add_argument("-o", "--output", help="CSV to write ('-' = steps table to stdout)")
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "previous_runs"), help="default: %(default)s")
    ap.add_argument("--all", action="store_true", help="every run under --runs-dir")
    ap.add_argument("--list", action="store_true", help="list the runs under --runs-dir and exit")
    ap.add_argument("--rounded", action="store_true", help="keep the 4-decimal console values, do not use log lines")
    args = ap.parse_args()

    if args.list:
        for d in all_runs(args.runs_dir):
            print("%s  %s" % (os.path.basename(d), " ".join(job_id_of(o) for o in outs_in(d))))
        return
    if args.all:
        outs = [o for d in all_runs(args.runs_dir) for o in outs_in(d)]
    elif args.runs:
        outs = [o for spec in args.runs for o in resolve(spec, args.runs_dir)]
    else:
        runs = all_runs(args.runs_dir)
        if not runs:
            sys.exit("collect_metrics: no run with a slurm-*.out under %s (pass a .out file or --runs-dir)" % args.runs_dir)
        outs = outs_in(runs[-1])
    outs = list(OrderedDict((os.path.abspath(o), True) for o in outs))
    if not outs:
        sys.exit("collect_metrics: nothing to read")

    step_rows, run_rows = [], []
    for out in outs:
        run_dir = os.path.dirname(out)
        run = os.path.basename(run_dir)
        job = job_id_of(out)
        rows, info = parse_out(out, rounded=args.rounded)
        for row in rows:
            full = OrderedDict([("run", run), ("job_id", job), ("step", row.get("trainer/global_step", ""))])
            full.update(row)
            step_rows.append(full)
        meta = OrderedDict([("run", run), ("job_id", job), ("out_file", out)])
        meta.update(info)
        meta.update(read_slurm_job(run_dir))
        meta.update(read_git(run_dir))
        meta.update(("cfg/" + k, v) for k, v in read_config(run_dir).items())
        run_rows.append(meta)
        sys.stderr.write("%s  job %s  %d steps\n" % (run, job or "?", len(rows)))

    if args.output == "-":
        columns = column_order(step_rows)
        w = csv.DictWriter(sys.stdout, fieldnames=columns, restval="", extrasaction="ignore")
        w.writeheader()
        for row in step_rows:
            w.writerow(row)
        return
    if args.output:
        steps_path = args.output
    elif len(outs) == 1:
        steps_path = os.path.join(os.path.dirname(outs[0]), "metrics.csv")
    else:
        steps_path = "metrics.csv"
    runs_path = os.path.splitext(steps_path)[0] + ".runs.csv"
    write_csv(steps_path, step_rows, column_order(step_rows))
    run_columns = list(OrderedDict((k, True) for r in run_rows for k in r))
    write_csv(runs_path, run_rows, run_columns)
    sys.stderr.write("wrote %s (%d rows, %d columns)\nwrote %s (%d runs)\n" % (
        steps_path, len(step_rows), len(column_order(step_rows)), runs_path, len(run_rows)))


if __name__ == "__main__":
    main()
