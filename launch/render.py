#!/usr/bin/env python3
"""
Turn a run config into the files a job reads. Called by submit.sh on the login node; needs only
PyYAML (cray-python has it, so does the container's /opt/venv). Python >= 3.6.

    render.py CONFIG [key.path=value ...]              write a new run dir, print its path
    render.py CONFIG [key.path=value ...] --dry-run    print the resolved config and SkyRL args only

Config files (configs/<task>/<setup>.yaml) are self-contained: one file = one setup, no includes.
    ${a.b.c}               value of another key in the same file. Typed when it is the whole string
                           (fsdp_size: ${derived.policy_gpus} stays an int), text otherwise.
    ${derived.x}           computed below (DERIVED); every value is listed in the run's config.yaml

Writes into <paths.runs>/<timestamp>-<config name>/:
    config.yaml            the fully resolved config, derived values included
    skyrl_args             one SkyRL override per line (key=value), passed to the entrypoint as-is
    job.env                variables for launch/job.sbatch and launch/harbor_env.sbatch
    env.sh                 the config's `env:` section, applied after launch/lumi_env.sh
    sbatch_args            sbatch flags of the training job (nodes = sum of layout nodes + judge.nodes)
    harbor_env.sbatch_args sbatch flags of the Harbor environment job (harbor configs only)
"""
import datetime
import getpass
import glob
import json
import os
import re
import shlex
import sys

try:
    import yaml
except ImportError:
    sys.exit("render.py: PyYAML missing. On a LUMI login node: module load cray-python")

SECTIONS = {"slurm", "paths", "model", "layout", "entrypoint", "env", "harbor_env", "judge", "skyrl"}
FREE_FORM = {"skyrl", "env"}  # command-line overrides may add new keys only here
REF = re.compile(r"\$\{([^}]+)\}")


class ConfigError(Exception):
    pass


# ---------------------------------------------------------------- loading
def load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - SECTIONS
    if unknown:
        raise ConfigError("{}: unknown top-level section(s) {}; allowed: {}".format(
            path, sorted(unknown), sorted(SECTIONS)))
    return cfg


def apply_override(cfg, item):
    if "=" not in item:
        raise ConfigError("override '{}' is not key.path=value".format(item))
    key, raw = item.split("=", 1)
    parts = key.split(".")
    if parts[0] not in SECTIONS:
        raise ConfigError("override '{}': unknown section '{}'".format(item, parts[0]))
    node = cfg
    for i, p in enumerate(parts[:-1]):
        if not isinstance(node.get(p), dict):
            if p in node or parts[0] not in FREE_FORM:
                raise ConfigError("override '{}': '{}' is not a section".format(item, ".".join(parts[:i + 1])))
            node[p] = {}
        node = node[p]
    if parts[-1] not in node and parts[0] not in FREE_FORM:
        raise ConfigError("override '{}': no such key (typo?). Only skyrl.* and env.* accept new keys".format(item))
    node[parts[-1]] = yaml.safe_load(raw) if raw != "" else ""


# ---------------------------------------------------------------- derived values
def model_source(r, key="model.path"):
    """Directory the nodes stage from: a local dir, or the snapshot of a hub id in hf_home."""
    path = r.get(key)
    if os.path.isdir(path):
        return path.rstrip("/")
    snaps = sorted(glob.glob(os.path.join(r.get("paths.hf_home"), "hub",
                                          "models--" + path.replace("/", "--"), "snapshots", "*")))
    if snaps:
        return snaps[0].rstrip("/")
    r.problem("{}={}: neither a directory nor a hub id under {}/hub{}".format(
        key, path, r.get("paths.hf_home"), SETUP_HINT))
    return path


def model_context(r):
    try:
        with open(os.path.join(r.derived("model_src"), "config.json")) as f:
            return int(json.load(f)["max_position_embeddings"])
    except Exception as e:  # noqa: BLE001
        r.problem("cannot read max_position_embeddings from {}/config.json ({}); assuming 32768".format(
            r.derived("model_src"), e))
        return 32768


def inference_gpus(r):
    return r.get("layout.inference_nodes") * r.get("layout.gpus_per_node")


def judge_model_source(r):
    if not r.derived("judge_nodes"):
        raise ConfigError("judge.nodes=0: the judge runs elsewhere (judge.base_url), no model is staged")
    return model_source(r, "judge.model.path")


JUDGE_ENTRYPOINT = "skyrl_extensions.judge.entrypoint"
SETUP_HINT = " (a fresh clone needs ./setup.sh first)"
# only meaningful when the config has the keys they read
OPTIONAL_DERIVED = {"max_input_tokens", "judge_served_name", "judge_model_src", "judge_model"}
DERIVED = {
    # layout
    "ray_nodes": lambda r: r.get("layout.policy_nodes") + r.get("layout.inference_nodes") + r.get("layout.ref_nodes"),
    "judge_nodes": lambda r: r.get("judge.nodes") if "judge" in r.cfg else 0,   # vLLM judge servers, outside Ray
    "num_nodes": lambda r: r.derived("ray_nodes") + r.derived("judge_nodes"),   # the Slurm allocation
    "policy_gpus": lambda r: r.get("layout.policy_nodes") * r.get("layout.gpus_per_node"),
    "ref_gpus": lambda r: r.get("layout.ref_nodes") * r.get("layout.gpus_per_node"),
    "inference_gpus": inference_gpus,
    "num_engines": lambda r: inference_gpus(r) // r.get("layout.tp") if inference_gpus(r) else 0,  # vLLM replicas
    "dp": lambda r: r.derived("policy_gpus") // r.get("layout.sp"),                               # FSDP data-parallel ranks
    # reference model: on its own nodes when layout.ref_nodes > 0, else on the policy GPUs
    "ref_colocated": lambda r: r.get("layout.ref_nodes") == 0,
    "ref_placement_nodes": lambda r: r.get("layout.ref_nodes") or r.get("layout.policy_nodes"),
    # model
    "model_src": model_source,
    "model": lambda r: "/tmp/{}-model/{}".format(getpass.getuser(), os.path.basename(r.get("model.path").rstrip("/"))),
    "served_name": lambda r: os.path.basename(r.get("model.path").rstrip("/")),  # no '/': Harbor addresses hosted_vllm/<name>
    "model_context": model_context,
    "max_model_len": lambda r: min(r.get("model.max_model_len_cap"), r.derived("model_context")),
    "max_input_tokens": lambda r: r.derived("max_model_len") - r.get("skyrl.generator.sampling_params.max_generate_length"),
    # judge (configs with a judge: section)
    "judge_served_name": lambda r: os.path.basename(r.get("judge.model.path").rstrip("/")),
    "judge_model_src": judge_model_source,
    "judge_model": lambda r: "/tmp/{}-judge/{}".format(getpass.getuser(), os.path.basename(r.get("judge.model.path").rstrip("/"))),
    # run
    "repo_dir": lambda r: os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # this clone = the default root
    "config_name": lambda r: r.config_name,
    "run_id": lambda r: "{}-{}".format(r.timestamp, r.config_name),
    "run_dir": lambda r: os.path.join(r.get("paths.runs"), r.derived("run_id")),
    "code_dir": lambda r: os.path.join(r.derived("run_dir"), "code"),  # snapshot of launch/, configs/, skyrl_extensions/
}


# ---------------------------------------------------------------- resolution
class Resolver:
    def __init__(self, cfg, config_name, strict):
        self.cfg = cfg
        self.config_name = config_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.strict = strict
        self.cache = {}
        self.stack = []

    def problem(self, msg):
        if self.strict:
            raise ConfigError(msg)
        print("render.py: warning: " + msg, file=sys.stderr)

    def derived(self, name):
        return self.get("derived." + name)

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        if key in self.stack:
            raise ConfigError("interpolation cycle: " + " -> ".join(self.stack + [key]))
        self.stack.append(key)
        try:
            if key.startswith("derived."):
                name = key[len("derived."):]
                if name not in DERIVED:
                    raise ConfigError("unknown ${{{}}}; available: {}".format(key, ", ".join(sorted(DERIVED))))
                value = DERIVED[name](self)
            else:
                node = self.cfg
                for p in key.split("."):
                    if not isinstance(node, dict) or p not in node:
                        raise ConfigError("${{{}}} is not set (referenced from {})".format(
                            key, self.stack[-2] if len(self.stack) > 1 else "the config"))
                    node = node[p]
                value = self.resolve(node, key)
        finally:
            self.stack.pop()
        self.cache[key] = value
        return value

    def resolve(self, value, where):
        if isinstance(value, dict):
            return {k: self.get(where + "." + k) for k in value}
        if isinstance(value, list):
            return [self.resolve(v, where) for v in value]
        if isinstance(value, str):
            m = REF.fullmatch(value)
            if m:
                return self.get(m.group(1).strip())
            return REF.sub(lambda m: scalar(self.get(m.group(1).strip())), value)
        return value

    def resolve_all(self):
        out = {section: self.get(section) for section in self.cfg}
        sandbox = self.get("paths.sandbox")
        if not os.path.exists(sandbox):
            self.problem("container paths.sandbox={} not found{}".format(sandbox, SETUP_HINT))
        out["derived"] = {}
        for name in DERIVED:
            try:
                out["derived"][name] = self.derived(name)
            except ConfigError:
                if name not in OPTIONAL_DERIVED:
                    raise
        return out


def iter_leaves(node, prefix):
    for k, v in node.items():
        key = prefix + k
        if isinstance(v, dict):
            for leaf in iter_leaves(v, key + "."):
                yield leaf
        else:
            yield key, v


# ---------------------------------------------------------------- formatting
def scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, list):
        return "[" + ",".join("'{}'".format(x) if isinstance(x, str) else scalar(x) for x in v) + "]"
    if isinstance(v, dict):
        raise ConfigError("a mapping cannot be inlined into a string: {}".format(v))
    return str(v)


def skyrl_args(skyrl):
    args = []
    for key, v in iter_leaves(skyrl, ""):
        s = scalar(v)
        if "\n" in s:
            raise ConfigError("skyrl.{}: value contains a newline".format(key))
        args.append("{}={}".format(key, s))
    return args


def lookup(cfg, key, default=None):
    node = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


# ---------------------------------------------------------------- checks
def check(c):
    """The layout and batch rules SkyRL would otherwise reject (or hang on) after the job started."""
    L, d = c["layout"], c["derived"]
    errors = []

    def need(ok, msg):
        if not ok:
            errors.append(msg)

    t = lambda k, default=None: lookup(c, "skyrl.trainer." + k, default)  # noqa: E731
    need(L["policy_nodes"] >= 1, "layout.policy_nodes must be >= 1")
    need(d["policy_gpus"] % L["sp"] == 0, "layout.sp={} must divide the {} policy GPUs".format(L["sp"], d["policy_gpus"]))
    need(d["ref_gpus"] % L["sp"] == 0, "layout.sp={} must divide the {} ref GPUs".format(L["sp"], d["ref_gpus"]))
    need(d["dp"] >= 2, "DP={}: lower layout.sp or raise layout.policy_nodes".format(d["dp"]))

    uses_engines = lookup(c, "skyrl.generator.inference_engine.num_engines") is not None
    if uses_engines:
        need(L["inference_nodes"] >= 1, "this config starts vLLM engines: layout.inference_nodes must be >= 1")
        need(L["inference_nodes"] == 0 or L["gpus_per_node"] % L["tp"] == 0,
             "layout.tp={} must divide gpus_per_node={} (an engine never spans nodes)".format(L["tp"], L["gpus_per_node"]))

    # batch rule (skyrl/train/utils/utils.py)
    train_bs, mini_bs = t("train_batch_size"), t("policy_mini_batch_size")
    n = lookup(c, "skyrl.generator.n_samples_per_prompt", 1)
    micro = t("micro_train_batch_size_per_gpu")
    if train_bs is not None and mini_bs is not None:
        need(train_bs % mini_bs == 0, "train_batch_size={} must be a multiple of policy_mini_batch_size={}".format(train_bs, mini_bs))
        need(mini_bs * n % d["dp"] == 0, "policy_mini_batch_size*n_samples_per_prompt={} not divisible by DP={}".format(mini_bs * n, d["dp"]))
        if micro is not None and mini_bs * n % d["dp"] == 0:
            per_rank = mini_bs * n // d["dp"]
            need(per_rank % micro == 0, "per-rank batch {} not divisible by micro_train_batch_size_per_gpu={}".format(per_rank, micro))

    gen = lookup(c, "skyrl.generator.sampling_params.max_generate_length")
    ctx = lookup(c, "skyrl.generator.inference_engine.engine_init_kwargs.max_model_len")
    if gen is not None and ctx is not None:
        need(gen < ctx, "max_generate_length={} must be below max_model_len={}".format(gen, ctx))

    h = c.get("harbor_env")
    conc = lookup(c, "skyrl.generator.rate_limit.max_concurrency")
    if h and h.get("submit") and conc is not None:
        workers = h["slurm"]["nodes"] * h["workers_per_node"]
        need(conc <= workers, "generator.rate_limit.max_concurrency={} exceeds the {} Harbor workers "
                              "(harbor_env.slurm.nodes x workers_per_node)".format(conc, workers))
    j = c.get("judge")
    judged = lookup(c, "skyrl.environment.env_class") == "judge"
    if j:
        need(j["nodes"] >= 0, "judge.nodes must be >= 0")
        need(bool(j["nodes"]) != bool(j.get("base_url")),
             "judge: set either nodes >= 1 (vLLM servers in this job) or base_url with nodes: 0 (a server that already runs)")
        need(L["gpus_per_node"] % j["tp"] == 0, "judge.tp={} must divide gpus_per_node={}".format(j["tp"], L["gpus_per_node"]))
        need(not j["nodes"] or d.get("judge_model_src"),
             "judge.model.path={}: neither a directory nor a hub id under {}/hub".format(
                 lookup(c, "judge.model.path"), c["paths"]["hf_home"]))
        need(j["name"] in judge_names(), "judge.name={}: not in skyrl_extensions/judge/judges.py ({})".format(
            j["name"], ", ".join(judge_names())))
        need(c["entrypoint"]["module"] == JUDGE_ENTRYPOINT,
             "a judge: section needs entrypoint.module={} (it registers the env)".format(JUDGE_ENTRYPOINT))
        need(not lookup(c, "skyrl.generator.batched", False),
             "generator.batched=true runs the judge requests one at a time: set it to false")
    else:
        need(not judged, "environment.env_class=judge needs a judge: section (configs/REFERENCE.yaml)")
    if errors:
        raise ConfigError("config checks failed:\n  " + "\n  ".join(errors))


def judge_names():
    """Keys of JUDGES in judges.py, read as text: this runs on the login node's Python, without pydantic."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skyrl_extensions", "judge", "judges.py")
    with open(path) as f:
        m = re.search(r"^JUDGES = \{(.*)\}", f.read(), re.M)
    return sorted(re.findall(r'"([^"]+)"\s*:', m.group(1))) if m else []


def summary(c):
    L, d = c["layout"], c["derived"]
    s = "layout: {}n | policy {}n/{} GCDs SP={} DP={} fsdp_size={}".format(
        d["num_nodes"], L["policy_nodes"], d["policy_gpus"], L["sp"], d["dp"],
        lookup(c, "skyrl.trainer.policy.fsdp_config.fsdp_size"))
    if not lookup(c, "skyrl.trainer.algorithm.use_kl_loss", False):
        s += " | no ref"
    else:
        s += " | ref " + ("on policy GPUs" if d["ref_colocated"] else "{}n".format(L["ref_nodes"]))
    if d["num_engines"]:
        s += " | vLLM {}n TP={} x {} engines".format(L["inference_nodes"], L["tp"], d["num_engines"])
    else:
        s += " | no vLLM"
    j = c.get("judge")
    if j and j["nodes"]:
        s += " | judge {}: {} on {}n TP={} x {} servers".format(
            j["name"], d["judge_served_name"], j["nodes"], j["tp"], j["nodes"] * L["gpus_per_node"] // j["tp"])
    elif j:
        s += " | judge {}: {} at {}".format(j["name"], d["judge_served_name"], j["base_url"])
    s += " | ctx {} | {} tasks x {} samples".format(
        d.get("max_model_len"), lookup(c, "skyrl.trainer.train_batch_size"),
        lookup(c, "skyrl.generator.n_samples_per_prompt", 1))
    return s


# ---------------------------------------------------------------- output files
def sbatch_flags(s, nodes, output, gpus_per_node=None):
    flags = ["--job-name=" + s["job_name"], "--account=" + s["account"], "--partition=" + s["partition"],
             "--nodes={}".format(nodes), "--time=" + str(s["time"]), "--cpus-per-task={}".format(s["cpus_per_task"]),
             "--output=" + output]
    if gpus_per_node:
        flags.append("--gpus-per-node={}".format(gpus_per_node))
    if s.get("mem") is not None:
        flags.append("--mem={}".format(s["mem"]))
    return flags + [str(x) for x in s.get("extra") or []]


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("".join(line + "\n" for line in lines))


def exports(values):
    return ["export {}={}".format(k, shlex.quote(scalar(v))) for k, v in values.items()]


def write_run_dir(c, args):
    d, p, L = c["derived"], c["paths"], c["layout"]
    run_dir = d["run_dir"]
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(c, f, default_flow_style=False, sort_keys=False, width=120)
    write_lines(os.path.join(run_dir, "skyrl_args"), args)

    h = c.get("harbor_env")
    job = {
        "RUN_DIR": run_dir,
        "CODE_DIR": d["code_dir"],  # snapshot made by submit.sh; on PYTHONPATH
        "SANDBOX": p["sandbox"],
        "BINDS": ",".join(p["binds"]),
        "HF_HOME": p["hf_home"],
        "GPUS_PER_NODE": L["gpus_per_node"],
        "NUM_NODES": d["num_nodes"],
        "RAY_PORT": L["ray_port"],
        "MODEL_SRC": d["model_src"],
        "MODEL": d["model"],
        "ENTRYPOINT": c["entrypoint"]["module"],
        "ENTRYPOINT_CWD": c["entrypoint"].get("cwd") or "",
        "LAYOUT_SUMMARY": summary(c),
        "HARBOR": 1 if h else 0,
        "RAY_NODES": d["ray_nodes"],        # nodes that join Ray; the remaining JUDGE_NODES serve the judge
        "JUDGE_NODES": d["judge_nodes"],
    }
    j = c.get("judge")
    if j:
        job.update({   # read by launch/job.sbatch (role judge) and skyrl_extensions/judge/env.py
            "JUDGE_NAME": j["name"],
            "JUDGE_SERVED_NAME": d["judge_served_name"],
            "JUDGE_MODEL_SRC": d.get("judge_model_src", ""),
            "JUDGE_MODEL": d["judge_model"],
            "JUDGE_TP": j["tp"],
            "JUDGE_PORT": j["port"],
            "JUDGE_VLLM_ARGS": j.get("vllm_args") or "",
            "JUDGE_URL": j.get("base_url") or "",
            "JUDGE_TEMPERATURE": j["temperature"],
            "JUDGE_MAX_TOKENS": j["max_tokens"],
            "JUDGE_MAX_CHARS": j["max_chars"],
            "JUDGE_EXTRA_BODY": json.dumps(j.get("extra_body") or {}),
            "JUDGE_TIMEOUT_S": j["timeout_s"],
            "JUDGE_STARTUP_TIMEOUT_S": j["startup_timeout_s"],
        })
    if h:
        job.update({
            "BRIDGE_FILE": h["bridge_file"],
            "HARBOR_WORKERS_PER_NODE": h["workers_per_node"],
            "HARBOR_BRIDGE_PORT": h["bridge_port"],
            "HARBOR_APPTAINER_DIR": h["apptainer_dir"],
            "HARBOR_SIF_CACHE": h["sif_cache"],
            "SINGULARITY_CACHEDIR": h["singularity_cachedir"],
            "BRIDGE_USE_FAKEROOT": h["use_fakeroot"],
        })
    write_lines(os.path.join(run_dir, "job.env"), ["# generated by launch/render.py"] + exports(job))
    write_lines(os.path.join(run_dir, "env.sh"), ["# config `env:` section, applied after launch/lumi_env.sh"]
                + exports(c.get("env") or {}))
    write_lines(os.path.join(run_dir, "sbatch_args"),
                sbatch_flags(c["slurm"], d["num_nodes"], os.path.join(run_dir, "slurm-%j.out"), L["gpus_per_node"]))
    if h and h["submit"]:
        hs = dict(c["slurm"], **h["slurm"])
        hs["job_name"] = h["slurm"].get("job_name", "harbor-env-" + d["config_name"])
        hs["extra"] = h["slurm"].get("extra", [])
        write_lines(os.path.join(run_dir, "harbor_env.sbatch_args"),
                    sbatch_flags(hs, hs["nodes"], os.path.join(run_dir, "harbor_env-%j.out")))
    return run_dir


def main():
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    argv = [x for x in argv if x != "--dry-run"]
    if not argv or argv[0] in ("-h", "--help"):
        sys.exit(__doc__)
    try:
        cfg = load(argv[0])
        for o in argv[1:]:
            apply_override(cfg, o)
        name = os.path.splitext(os.path.basename(argv[0]))[0]
        c = Resolver(cfg, name, strict=not dry_run).resolve_all()
        check(c)
        args = skyrl_args(c["skyrl"])
        print(summary(c), file=sys.stderr)
        if dry_run:
            yaml.safe_dump(c, sys.stdout, default_flow_style=False, sort_keys=False, width=120)
            print("# ---- SkyRL args ({} on the command line of python -m {})".format(len(args), c["entrypoint"]["module"]))
            print("\n".join(args))
            return
        print(write_run_dir(c, args))
    except ConfigError as e:
        sys.exit("render.py: " + str(e))


if __name__ == "__main__":
    main()
