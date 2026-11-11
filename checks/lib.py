"""
Shared pieces of the check suite: the result type, the registry, the context a check receives,
and a small result cache. See checks/runner.py for how checks are run.

A check is a function `fn(ctx) -> Result` registered with @check("<stage>", "<area>.<name>").
It returns PASS / WARN / FAIL / SKIP with a one-line summary and optional detail lines.
"""
import hashlib
import json
import os
import time

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
STAGES = ("pre", "run", "export")
REGISTRY = []  # (stage, name, fn, cache_inputs)


class Result:
    def __init__(self, status, summary, details=None, data=None):
        self.status, self.summary = status, summary
        self.details = list(details or [])
        self.data = data or {}  # numbers behind the verdict, for --json

    def as_dict(self):
        return {"status": self.status, "summary": self.summary, "details": self.details, "data": self.data}


def check(stage, name, cache_inputs=None):
    """
    cache_inputs: optional `fn(ctx) -> list of file paths`. When given, a PASS is stored under
    <root>/.check_cache keyed by the check's source file and the content of these files, and reused
    until any of them changes. Use it for checks that are slow to start (they import vLLM).
    """
    assert stage in STAGES, stage

    def deco(fn):
        REGISTRY.append((stage, name, fn, cache_inputs))
        return fn

    return deco


class Ctx:
    """What a check may read. Fields that a stage does not have are None."""

    def __init__(self, root, cfg=None, model_dir=None, run_dir=None, log=None, export_dir=None, base_dir=None, opts=None):
        self.root = root              # repo root = working root
        self.cfg = cfg                # resolved config (dict), stage pre
        self.model_dir = model_dir    # stage pre: derived.model_src
        self.run_dir, self.log = run_dir, log              # stage run
        self.export_dir, self.base_dir = export_dir, base_dir   # stage export
        self.opts = opts or {}

    def get(self, path, default=None):
        """ctx.get("skyrl.trainer.algorithm.use_kl_loss")"""
        node = self.cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


# ---------------------------------------------------------------- cache
def _digest(paths, source_file):
    h = hashlib.sha256()
    for p in [source_file] + sorted(paths):
        h.update(p.encode())
        try:
            st = os.stat(p)
            if st.st_size <= 64 << 20:          # small file: content
                with open(p, "rb") as f:
                    h.update(f.read())
            else:                               # large file (parquet, weights): size + mtime
                h.update("{}:{}".format(st.st_size, int(st.st_mtime)).encode())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:24]


def cached_run(ctx, name, fn, cache_inputs, use_cache=True):
    if cache_inputs is None or not use_cache:
        return fn(ctx)
    cache_dir = os.path.join(ctx.root, ".check_cache")
    key = _digest(cache_inputs(ctx), fn.__code__.co_filename)
    path = os.path.join(cache_dir, "{}-{}.json".format(name, key))
    if os.path.exists(path):
        d = json.load(open(path))
        return Result(d["status"], "{}  (cached {})".format(d["summary"], d["when"]), d["details"], d["data"])
    res = fn(ctx)
    if res.status == PASS:  # only a PASS is reused; anything else is recomputed until it is fixed
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(dict(res.as_dict(), when=time.strftime("%Y-%m-%d")), open(path, "w"))
    return res


def small_model_files(model_dir):
    """Everything in a model dir except the weight shards: the files that decide how weights are read."""
    return [os.path.join(model_dir, f) for f in sorted(os.listdir(model_dir))
            if not f.endswith((".safetensors", ".bin", ".pt")) and os.path.isfile(os.path.join(model_dir, f))]
