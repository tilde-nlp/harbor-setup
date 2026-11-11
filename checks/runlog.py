"""
Checks on a run's Slurm log: is the run trustworthy, from step 1 on?

SkyRL prints one metrics dict per step ('policy/grad_norm': '0.1356', ...). parse() turns the log
into a list of {metric: value} per step; the checks read that list. Works on a finished or a
running job (run it after step 1 to catch a broken run early).
"""
import glob
import math
import os
import re

from checks.lib import FAIL, PASS, SKIP, WARN, Result, check

# Thresholds, with the measurements they come from (notes/TRAINING_SANITY_CHECK.md):
NOISE_FLOOR_MAX = 0.05       # pregen step 1, policy == reference: measured 0.011 (bs40), 0.024 (bs5)
NOISE_CLIP_MAX = 0.05        # share of tokens outside the ratio band at step 1: measured 0.0065, 0.012
ROLLOUT_MISMATCH_MAX = 0.05  # online, trainer vs vLLM: Qwen 0.008-0.011; TildeOpen with the RoPE bug 0.10-0.21
DRIFT_CLIP_MAX = 0.20        # late in a run: share of tokens outside the band (bs5 run ended at ~0.20)
DRIFT_KL_FACTOR = 10.0       # policy KL relative to its step-1 value

ABS_DIFF = "policy/minibatch_rollout_logprobs_abs_diff_mean"
ROLLOUT_DIFF = "policy/rollout_train_logprobs_abs_diff_mean"
CLIP = "policy/loss_metrics/clip_ratio"
KL = "policy/policy_kl"
STEP = "trainer/global_step"

_METRIC = re.compile(r"'([A-Za-z0-9_./@-]+)': '?(-?(?:[0-9][0-9.eE+-]*|nan|inf))'?[,}]")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ERROR = re.compile(r"\| ERROR +\||FATAL|out of memory|OutOfMemory|HIP error|NCCL WARN|Training failed|RayTaskError|"
                    r"CANCELLED|DUE TO TIME LIMIT|^\d+: (Traceback|[A-Za-z_.]+Error:)")
_NOT_ERROR = re.compile(r"<\|tool_output|could not parse the tool call")   # tracebacks quoted inside example rollouts
_PROGRESS = re.compile(r"Training Batches Processed: +\d+%\|[^|]*\| *(\d+)/(\d+)")


def find_log(path):
    if os.path.isfile(path):
        return path
    logs = sorted(glob.glob(os.path.join(path, "slurm-*.out")) + glob.glob(os.path.join(path, "*.out")), key=os.path.getmtime)
    if not logs:
        raise FileNotFoundError("no slurm-*.out or *.out in {}".format(path))
    return logs[-1]


def parse(log_path):
    """-> (steps, errors, progress): steps = [{metric: float}], errors = [log lines], progress = (done, total) or None"""
    steps, cur, errors, progress = [], {}, [], None
    with open(log_path, errors="replace") as f:
        for raw in f:
            line = _ANSI.sub("", raw)
            m = _METRIC.search(line)
            if m:
                cur[m.group(1)] = float(m.group(2))
                if m.group(1) == STEP:
                    steps.append(cur)
                    cur = {}
                continue
            p = _PROGRESS.search(line)
            if p:
                progress = (int(p.group(1)), int(p.group(2)))
            if _ERROR.search(line) and not _NOT_ERROR.search(line):
                errors.append(line.strip()[:200])
    return steps, errors, progress


def is_pregen(steps):
    return any(k.startswith("generate/") and ("replay" in k or "pregenerated" in k) for k in steps[0])


def table(steps, every=1):
    cols = [("step", STEP, "{:.0f}"), ("abs_diff", ABS_DIFF, "{:.4f}"), ("roll_diff", ROLLOUT_DIFF, "{:.4f}"), ("clip", CLIP, "{:.4f}"),
            ("kl", KL, "{:.4f}"), ("gnorm", "policy/grad_norm", "{:.3f}"), ("entropy", "policy/policy_entropy", "{:.3f}"),
            ("reward", "reward/avg_raw_reward", "{:.3f}"), ("step_s", "timing/step", "{:.0f}")]
    cols = [c for c in cols if any(c[1] in s for s in steps)]
    lines = [" ".join("{:>9s}".format(c[0]) for c in cols)]
    for i, s in enumerate(steps):
        if i % every == 0 or i == len(steps) - 1:
            lines.append(" ".join("{:>9s}".format(c[2].format(s[c[1]]) if c[1] in s else "-") for c in cols))
    return lines


def _steps(ctx):
    if "parsed" not in ctx.opts:
        ctx.opts["parsed"] = parse(ctx.log)
    return ctx.opts["parsed"]


@check("run", "run.steps")
def run_steps(ctx):
    """Which steps the log covers, and whether the run is pregenerated or online."""
    steps, _, progress = _steps(ctx)
    if not steps:
        return Result(FAIL, "no step has finished yet (no metrics in {})".format(os.path.basename(ctx.log)))
    first, last = int(steps[0][STEP]), int(steps[-1][STEP])
    kind = "pregenerated" if is_pregen(steps) else "online"
    total = " of {}".format(progress[1]) if progress else ""
    return Result(PASS, "{} run, steps {}..{}{} in this log".format(kind, first, last, total))


@check("run", "run.step1_noise_floor")
def step1_noise_floor(ctx):
    """Pregenerated runs: at step 1 the policy equals the reference, so every importance ratio should be 1."""
    steps = _steps(ctx)[0]
    if not steps or not is_pregen(steps):
        return Result(SKIP, "not a pregenerated run")
    s = steps[0]
    if int(s[STEP]) != 1:
        return Result(SKIP, "log starts at step {} (resumed): step 1 is in an earlier log".format(int(s[STEP])))
    if ABS_DIFF not in s:
        return Result(WARN, "{} not logged at step 1".format(ABS_DIFF))
    d, c = s[ABS_DIFF], s.get(CLIP, 0.0)
    line = "step 1: policy vs reference |log-prob diff| {:.4f}, {:.2%} of tokens outside the ratio band".format(d, c)
    if d > NOISE_FLOOR_MAX or c > NOISE_CLIP_MAX:
        return Result(FAIL, line, ["identical weights should give ~0.01-0.02 and ~1% (bf16 noise); more means the policy and "
                                   "reference workers do not compute the same network (config, packing, SP, dtype)"])
    return Result(PASS, line + "  (this is the run's noise floor)")


@check("run", "run.rollout_mismatch")
def rollout_mismatch(ctx):
    """Online runs: do the trainer and the vLLM engines agree on the log-probs of the sampled tokens?"""
    steps = _steps(ctx)[0]
    if not steps or is_pregen(steps):
        return Result(SKIP, "not an online run")
    key = ROLLOUT_DIFF if any(ROLLOUT_DIFF in s for s in steps) else ABS_DIFF
    vals = [s[key] for s in steps[:3] if key in s]
    if not vals:
        return Result(WARN, "no rollout log-prob diff in the log: set generator.sampling_params.logprobs=0 to get it")
    d = sum(vals) / len(vals)
    line = "trainer vs vLLM |log-prob diff| {:.4f} over the first {} step(s)".format(d, len(vals))
    if d > ROLLOUT_MISMATCH_MAX:
        return Result(FAIL, line, ["a healthy stack shows ~0.01; this run trains on samples from a different network than it optimises",
                                   "run ./check.sh pre <run yaml> (model.rope_parity) and ./check.sh gpu <run yaml>"])
    return Result(PASS, line)


@check("run", "run.errors")
def run_errors(ctx):
    """NaN/inf metrics and error lines (OOM, NCCL, time limit, tracebacks)."""
    steps, errors, _ = _steps(ctx)
    bad = sorted({k for s in steps for k, v in s.items() if math.isnan(v) or math.isinf(v)})
    if bad:
        return Result(FAIL, "NaN/inf in {} metric(s)".format(len(bad)), bad[:10] + errors[:5])
    if errors:
        return Result(FAIL, "{} error line(s) in the log".format(len(errors)), errors[:8])
    return Result(PASS, "no NaN/inf metric, no error line")


@check("run", "run.drift")
def drift(ctx):
    """Late in the run: share of tokens outside the ratio band and policy KL, relative to step 1."""
    steps = _steps(ctx)[0]
    if len(steps) < 2:
        return Result(SKIP, "fewer than 2 steps")
    notes = []
    clips = [s[CLIP] for s in steps if CLIP in s]
    kls = [s[KL] for s in steps if KL in s]
    tail = max(1, len(steps) // 10)
    if clips and sum(clips[-tail:]) / tail > DRIFT_CLIP_MAX:
        notes.append("{:.0%} of tokens are outside the ratio band over the last {} steps (step 1: {:.1%}): they get no gradient".format(
            sum(clips[-tail:]) / tail, tail, clips[0]))
    if kls and kls[0] > 0 and sum(kls[-tail:]) / tail > DRIFT_KL_FACTOR * kls[0]:
        notes.append("policy KL to the reference is {:.4f} over the last {} steps, {:.0f}x its first value {:.4f}".format(
            sum(kls[-tail:]) / tail, tail, sum(kls[-tail:]) / tail / kls[0], kls[0]))
    if notes:
        return Result(WARN, "the policy has moved far from the reference", notes)
    return Result(PASS, "clip ratio {} and policy KL {} stay near their first values".format(
        "{:.3f}".format(clips[-1]) if clips else "n/a", "{:.4f}".format(kls[-1]) if kls else "n/a"))
