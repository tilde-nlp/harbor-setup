"""
Checks on an HF export, before anyone evaluates it.

    <export dir>/policy        what SkyRL wrote (fp32): start weights + the trained update
    <export dir>/policy_bf16   optional bf16 copy (hf_to_bf16.py); when absent, the cast is simulated

The start checkpoint is stored in bf16 and trained in fp32. A small update does not survive the
cast back to bf16: each weight either snaps back to its start value or jumps a whole bf16 step.
Measured (notes/TRAINING_SANITY_CHECK.md, section 3): cosine 0.52 after 162 steps at lr 1e-6,
0.77 after 1296 steps.
"""
import collections
import os

import numpy as np

from checks.lib import FAIL, PASS, SKIP, WARN, Result, check
from checks.st import Model, round_bf16

MIN_COSINE = 0.9       # cosine(true update, update in the bf16 copy); 1.0 = the copy carries the update
SAMPLE = 100000        # elements read per tensor (from its start)
MAX_TENSORS = 120      # tensors read, evenly spread over the model (all of them: opts["all_tensors"])
MAX_RELATIVE_UPDATE = 0.01   # |update| / |weights|. The pregenerated runs: 0.0005-0.0015. Above this the result is only
                             # a WARN: either a much stronger training, or --base is not the checkpoint the run
                             # started from (two SFT checkpoints 14k steps apart: cosine 1.000, a false "all fine")


def group(key):
    for name, tag in (("norm", "norm weights"), ("embed_tokens", "embed_tokens"), ("lm_head", "lm_head"),
                      ("self_attn", "attention proj"), ("mlp", "mlp proj")):
        if name in key:
            return tag
    return "other"


def delta_stats(base_dir, fp32_dir, bf16_dir=None, sample=SAMPLE, max_tensors=MAX_TENSORS):
    """-> {group: dict(n, changed, cos, kept, rms_update, cast_mismatch)} including "ALL"; raises ValueError on dtype surprises"""
    base, fp32 = Model(base_dir), Model(fp32_dir)
    bf16 = Model(bf16_dir) if bf16_dir else None
    if set(base.map) != set(fp32.map):
        raise ValueError("key sets differ: {} only in base, {} only in export".format(
            sorted(set(base.map) - set(fp32.map))[:3], sorted(set(fp32.map) - set(base.map))[:3]))
    keys = list(fp32.map)
    if max_tensors and len(keys) > max_tensors:
        picked = {keys[int(i)] for i in np.linspace(0, len(keys) - 1, max_tensors)}
        picked |= {k for k in keys if "norm" in k and "layers" not in k}
        keys = [k for k in keys if k in picked]
    acc = collections.defaultdict(lambda: np.zeros(7))  # n, changed, d.dh, d.d, dh.dh, cast mismatches, b.b
    for k in keys:
        b, _ = base.read(k, sample)
        f, ft = fp32.read(k, sample)
        if ft != "F32":
            raise ValueError("{}: the export is {}, not fp32: the update cannot be separated from rounding".format(k, ft))
        h_calc = round_bf16(f)
        h = bf16.read(k, sample)[0] if bf16 else h_calc
        d, dh = f - b, h - round_bf16(b)
        acc[group(k)] += [len(b), (dh != 0).sum(), (d * dh).sum(), (d * d).sum(), (dh * dh).sum(), (h != h_calc).sum(), (b * b).sum()]
    out = {}
    acc["ALL"] = sum(acc.values())
    for g, (n, ch, ddh, dd, dhdh, mis, bb) in acc.items():
        out[g] = dict(n=int(n), changed=float(ch / n), cos=float(ddh / max(np.sqrt(dd * dhdh), 1e-300)),
                      kept=float(np.sqrt(dhdh / max(dd, 1e-300))), rms_update=float(np.sqrt(dd / n)), cast_mismatch=int(mis),
                      relative_update=float(np.sqrt(dd / max(bb, 1e-300))))
    return out


def export_dirs(export_dir):
    """-> (fp32 dir, bf16 dir or None). Accepts .../global_step_N or .../global_step_N/policy."""
    p = os.path.join(export_dir, "policy")
    fp32 = p if os.path.isdir(p) else export_dir
    b = os.path.join(os.path.dirname(fp32.rstrip("/")), "policy_bf16")
    return fp32, (b if os.path.isdir(b) else None)


@check("export", "export.delta")
def export_delta(ctx):
    """How much of the trained update a bf16 copy of the export keeps (cosine with the true update)."""
    fp32, bf16 = export_dirs(ctx.export_dir)
    try:
        s = delta_stats(ctx.base_dir, fp32, bf16, max_tensors=None if ctx.opts.get("all_tensors") else MAX_TENSORS)
    except ValueError as e:
        return Result(FAIL, str(e))
    a = s["ALL"]
    rows = ["{:15s} changed {:6.2%}  cosine {:.3f}  kept {:.3f}  rms update {:.2e}".format(g, v["changed"], v["cos"], v["kept"], v["rms_update"])
            for g, v in sorted(s.items())]
    if a["rms_update"] == 0:
        return Result(FAIL, "the export is identical to the start checkpoint: nothing was trained or the wrong directory was given", data=s)
    what = "the bf16 copy" if bf16 else "a bf16 cast"
    line = "{} holds the update with cosine {:.3f} ({:.1%} of weights change at all)".format(what, a["cos"], a["changed"])
    if a["cast_mismatch"]:
        return Result(FAIL, "{} values of policy_bf16 are not round(policy): the copy was not made from this export".format(a["cast_mismatch"]), rows, s)
    if a["cos"] < MIN_COSINE:
        return Result(FAIL, line, rows + ["about {:.0%} of what differs from the start model in bf16 is the trained update, the rest is rounding".format(a["cos"] ** 2),
                                         "evaluate the fp32 directory instead: {}".format(fp32)], s)
    if a["relative_update"] > MAX_RELATIVE_UPDATE:
        return Result(WARN, line + ", but the export is {:.1%} of the weight norm away from --base".format(a["relative_update"]),
                      ["the pregenerated runs moved the weights by 0.05-0.15%. Is --base the checkpoint this run started from "
                       "(the run's model.path)? With another checkpoint as base this check always looks fine."], s)
    return Result(PASS, line + "; update is {:.2%} of the weight norm".format(a["relative_update"]), data=s)


@check("export", "export.rope")
def export_rope(ctx):
    """The export's config must give the same rotary tables as the start model's, in both libraries."""
    from checks.model import TOLERANCE, load_config, rope_tables_diff

    fp32, _ = export_dirs(ctx.export_dir)
    try:
        d_e, h_e, v_e = rope_tables_diff(load_config(fp32))
        _, h_b, v_b = rope_tables_diff(load_config(ctx.base_dir))
    except StopIteration:
        return Result(SKIP, "model type has no rotary embedding class in transformers")
    line = "export: transformers x{:.4f}, vLLM x{:.4f}; start model: x{:.4f}, x{:.4f}".format(h_e, v_e, h_b, v_b)
    if d_e > TOLERANCE:
        return Result(FAIL, line + ": the export's config makes the two libraries disagree",
                      ["fix the export's config.json the way the start model's was fixed (notes/ROPE_SCALE_MISMATCH.md)"])
    if abs(h_e - h_b) > 1e-6 or abs(v_e - v_b) > 1e-6:
        return Result(FAIL, line + ": the export would run at a different RoPE magnitude than it was trained with")
    return Result(PASS, line)


@check("export", "export.tokenizer")
def export_tokenizer(ctx):
    """The tokenizer files SkyRL wrote into the export must tokenize like the start model's golden file."""
    from checks.tokenizer import GOLDEN, golden_result

    fp32, bf16 = export_dirs(ctx.export_dir)
    golden = os.path.join(ctx.base_dir, GOLDEN)
    results = [golden_result(d, golden, "{}'s tokenizer".format(os.path.basename(d))) for d in (fp32, bf16) if d]
    worst = max(results, key=lambda r: [PASS, SKIP, WARN, FAIL].index(r.status))
    if worst.status == FAIL:
        worst.details.append("replace the export's tokenizer files (tokenizer.json, tokenizer_config.json, chat_template.jinja) "
                             "with the start model's tokenizer files before evaluating")
    return worst
