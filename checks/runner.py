#!/usr/bin/env python
"""
Run the checks of one stage and print a verdict per check. Normally called through ./check.sh.

    python -m checks.runner pre    <run.yaml> [key.path=value ...]   before submitting (./submit.sh does this)
    python -m checks.runner run    <run dir | slurm log> [--table N] after step 1 and at the end of a run
    python -m checks.runner export <export dir> --base <model dir>   before evaluating an export
    python -m checks.runner list                                     every check, by stage
    python -m checks.runner model-dir <run.yaml> [overrides]         print the model directory of a run

Options: --no-cache (recompute cached PASSes), --json FILE (machine-readable results),
         --only SUBSTR (run the checks whose name contains SUBSTR), --all-tensors (export: read every tensor).
Exit code 1 if any check FAILs. WARN does not fail the stage: read it.
"""
import argparse
import json
import os
import subprocess
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checks import model, tokenizer, data, settings, runlog, export  # noqa: I001,E402,F401  (importing registers the checks)
from checks.lib import FAIL, PASS, REGISTRY, SKIP, WARN, Ctx, Result, cached_run  # noqa: E402

ORDER = {FAIL: 0, WARN: 1, PASS: 2, SKIP: 3}


def resolve_run(argv):
    """The fully resolved run yaml, exactly as ./submit.sh would use it (launch/render.py --dry-run)."""
    import yaml

    p = subprocess.run([sys.executable, os.path.join(ROOT, "launch", "render.py")] + argv + ["--dry-run"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode != 0:
        sys.exit(p.stderr.strip() or "render.py failed")
    return yaml.safe_load(p.stdout.split("\n# ---- SkyRL args")[0])


def run_stage(stage, ctx, use_cache=True, only=None):
    results = []
    for st, name, fn, cache_inputs in REGISTRY:
        if st != stage or (only and only not in name):
            continue
        try:
            res = cached_run(ctx, name, fn, cache_inputs, use_cache)
        except Exception as e:  # a crashing check must not look like a pass
            res = Result(FAIL, "the check itself crashed: {}: {}".format(type(e).__name__, str(e)[:200]),
                         traceback.format_exc().strip().splitlines()[-6:])
        results.append((name, res))
    return results


def report(results, out=sys.stdout):
    width = max(len(n) for n, _ in results)
    for name, r in results:
        print("{:4s}  {:{w}s}  {}".format(r.status, name, r.summary, w=width), file=out)
    for name, r in results:
        if r.status in (FAIL, WARN) and r.details:
            print("\n{} {}".format(r.status, name), file=out)
            for d in r.details:
                if d:
                    print("    " + str(d), file=out)
    counts = {s: sum(1 for _, r in results if r.status == s) for s in ORDER}
    print("\n" + ", ".join("{} {}".format(counts[s], s) for s in sorted(ORDER, key=ORDER.get) if counts[s]), file=out)
    return counts[FAIL] == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pre", "run", "export", "list", "model-dir"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--base")
    ap.add_argument("--table", type=int, default=0, metavar="N", help="run: print the per-step table, every N-th step")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--all-tensors", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.stage == "list":
        for stage in ("pre", "run", "export"):
            print(stage)
            for st, name, fn, _ in REGISTRY:
                if st == stage:
                    print("  {:28s} {}".format(name, (fn.__doc__ or "").strip().splitlines()[0]))
        return
    if not args.target:
        ap.error("{} needs a target".format(args.stage))

    if args.stage in ("pre", "model-dir"):
        cfg = resolve_run([args.target] + args.overrides)
        model_dir = cfg["derived"]["model_src"]
        if args.stage == "model-dir":
            print(model_dir)
            return
        ctx = Ctx(ROOT, cfg=cfg, model_dir=model_dir, opts={"uses_vllm": bool(cfg["derived"].get("num_engines"))})
        print("pre-submit checks for {}   (model dir {})\n".format(args.target, model_dir))
    elif args.stage == "run":
        log = runlog.find_log(args.target)
        ctx = Ctx(ROOT, run_dir=args.target if os.path.isdir(args.target) else os.path.dirname(log), log=log)
        print("run checks on {}\n".format(log))
    else:
        if not args.base:
            ap.error("export needs --base <the model directory training started from>")
        ctx = Ctx(ROOT, export_dir=args.target.rstrip("/"), base_dir=args.base.rstrip("/"), opts={"all_tensors": args.all_tensors})
        print("export checks on {}   (start model {})\n".format(ctx.export_dir, ctx.base_dir))

    results = run_stage(args.stage, ctx, use_cache=not args.no_cache, only=args.only)
    if not results:
        sys.exit("no check matches")
    if args.stage == "run" and args.table and ctx.opts.get("parsed", ([],))[0]:
        print("\n".join(runlog.table(ctx.opts["parsed"][0], args.table)) + "\n")
    ok = report(results)
    if args.json:
        json.dump({n: r.as_dict() for n, r in results}, open(args.json, "w"), indent=1, default=str)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
