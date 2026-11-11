"""Checks on the resolved run settings that render.py's layout/batch rules do not cover."""
import glob
import json
import os

import numpy as np

from checks.lib import FAIL, PASS, SKIP, WARN, Result, _digest, check, small_model_files
from checks.st import Model, bf16_step

# rms weight update per (lr x optimizer steps), measured on the two pregenerated runs (AdamW, warm-up 10):
# 1.13e-5 / (1e-6 x 162) = 0.07 and 3.07e-5 / (1e-6 x 1296) = 0.024. The smaller one is used. With the
# median bf16 step of TildeOpen (6.1e-5) those runs had update/step 0.19 and 0.50 and kept the update
# with cosine 0.52 and 0.77: the check asks for update/step >= 1.
UPDATE_PER_LR_STEP = 0.024


@check("pre", "settings.ref_temperature")
def ref_temperature(ctx):
    """The reference worker ignores sampling temperature: anything but 1.0 breaks KL and importance ratios."""
    temp = ctx.get("skyrl.generator.sampling_params.temperature", 1.0)
    if not ctx.get("skyrl.trainer.algorithm.use_kl_loss", True):
        return Result(SKIP, "no reference worker (use_kl_loss=false)")
    if temp in (None, 1, 1.0):
        return Result(PASS, "temperature 1.0 with a reference worker")
    return Result(FAIL, "temperature={} with use_kl_loss=true".format(temp),
                  ["the reference worker's forward does not apply the temperature (RefWorkerBase._forward_micro_batch), "
                   "the policy's does: KL and importance ratios compare two different distributions"])


@check("pre", "settings.export_dtype")
def export_dtype(ctx):
    """Will the trained update survive a cast of the export to bf16?  (notes/TRAINING_SANITY_CHECK.md, section 3)"""
    lr = ctx.get("skyrl.trainer.policy.optimizer_config.lr")
    steps = ctx.get("skyrl.trainer.max_training_steps")
    if lr is None or not steps:
        return Result(SKIP, "lr or trainer.max_training_steps not set: cannot estimate the size of the update")
    try:
        m = Model(ctx.model_dir)
        keys = [k for k in m.map if k.endswith("weight") and "mlp" in k] or list(m.map)
        w, _ = m.read(keys[len(keys) // 2], 200000)
    except (OSError, KeyError, ValueError) as e:  # hub ids, .bin checkpoints
        return Result(SKIP, "cannot read weights from {} ({})".format(ctx.model_dir, type(e).__name__))
    step = float(np.median(bf16_step(w[w != 0])))
    update = UPDATE_PER_LR_STEP * float(lr) * int(steps)
    data = {"bf16_step": step, "expected_rms_update": update}
    msg = "expected weight update ~{:.1e} (lr {} x {} steps), bf16 step at a typical weight {:.1e}".format(update, lr, steps, step)
    if update >= step:
        return Result(PASS, msg, data=data)
    return Result(WARN, msg + ": a bf16 copy of the result will lose most of the update",
                  ["evaluate the fp32 export, and run `./check.sh export <export dir>` before any evaluation",
                   "measured: with update/step 0.19 the bf16 copy held 27% of the update, with 0.50 it held 59%"], data)


def gpu_record_path(root, model_dir):
    """One record per state of the model dir's small files: changing them invalidates it."""
    return os.path.join(root, ".check_cache", "gpu_parity-{}.json".format(_digest(small_model_files(model_dir), "gpu_parity")))


@check("pre", "settings.gpu_parity_record")
def gpu_parity_record(ctx):
    """Has the one-node vLLM-vs-transformers log-prob comparison been run for this model directory?"""
    path = gpu_record_path(ctx.root, ctx.model_dir)
    if not os.path.exists(path):
        old = glob.glob(os.path.join(ctx.root, ".check_cache", "gpu_parity-*.json"))
        return Result(WARN, "no GPU parity record for this model directory{}".format(" (its files changed since the last one)" if old else ""),
                      ["one node, ~20 min: ./check.sh gpu <run yaml>   (compares per-token log-probs of vLLM and transformers)"])
    d = json.load(open(path))
    line = "vLLM vs transformers mean |diff| {:.4f}, mean signed {:+.4f} nats/token (job {}, {})".format(
        d["mean_abs_diff"], d["mean_signed_diff"], d.get("job", "?"), d.get("when", "?"))
    return Result(PASS if d["status"] == "PASS" else FAIL, line, data=d)
