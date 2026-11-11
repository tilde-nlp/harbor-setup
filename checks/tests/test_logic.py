"""
Tests of the checks' logic on synthetic inputs: no model, no GPU, no vLLM. Run anywhere with
numpy + pyarrow + pytest:   python -m pytest checks/tests/test_logic.py
"""
import numpy as np
import pytest

from checks import data, export, model, runlog, settings
from checks.lib import FAIL, PASS, SKIP, WARN, Ctx
from checks.st import bf16_step, round_bf16, write_safetensors


# ---------------------------------------------------------------- bf16 arithmetic and export.delta
def test_bf16_rounding_and_step():
    assert bf16_step(0.016) == pytest.approx(2.0 ** -13)            # 0.016 is in [2^-6, 2^-5): step 2^-6 / 128
    grid = round_bf16(np.array([0.016]))[0]
    assert round_bf16(np.array([grid + 0.1 * 2.0 ** -13]))[0] == grid                 # a tenth of a step is erased
    assert round_bf16(np.array([grid + 0.6 * 2.0 ** -13]))[0] == grid + 2.0 ** -13    # more than half a step jumps a whole one


def _export(tmp_path, update_in_steps):
    rng = np.random.default_rng(0)
    base = round_bf16(rng.normal(0, 0.016, 50000))
    trained = base + rng.normal(0, 1, base.size) * update_in_steps * bf16_step(base)
    for name, arr, dtype in (("base", base, "BF16"), ("policy", trained, "F32"), ("policy_bf16", trained, "BF16")):
        (tmp_path / name).mkdir()
        write_safetensors(str(tmp_path / name / "model.safetensors"), {"model.layers.0.mlp.up_proj.weight": (arr, dtype)})
    return Ctx("/nonexistent", export_dir=str(tmp_path), base_dir=str(tmp_path / "base"))


def test_export_delta_fails_when_update_is_below_the_bf16_step(tmp_path):
    res = export.export_delta(_export(tmp_path, update_in_steps=0.1))
    assert res.status == FAIL and res.data["ALL"]["cos"] < 0.7
    assert any("fp32 directory" in d for d in res.details)


def test_export_delta_passes_when_update_is_well_above_the_bf16_step(tmp_path):
    res = export.export_delta(_export(tmp_path, update_in_steps=5.0))
    assert res.status in (PASS, WARN) and res.data["ALL"]["cos"] > 0.95   # WARN: 5 bf16 steps is > 1% of these weights


def test_export_delta_fails_on_untrained_export(tmp_path):
    res = export.export_delta(_export(tmp_path, update_in_steps=0.0))
    assert res.status == FAIL and "identical" in res.summary


def test_export_delta_warns_when_base_is_probably_not_the_start_checkpoint(tmp_path):
    ctx = _export(tmp_path, update_in_steps=5.0)
    other = round_bf16(np.random.default_rng(1).normal(0, 0.016, 50000))      # a different checkpoint of the same model
    write_safetensors(str(tmp_path / "base" / "model.safetensors"), {"model.layers.0.mlp.up_proj.weight": (other, "BF16")})
    res = export.export_delta(ctx)
    assert res.status == WARN and "away from --base" in res.summary


# ---------------------------------------------------------------- data.pregen
def _parquet(tmp_path, mutate=None, n=4, groups=6):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for g in range(groups):
        for r in range(n):
            resp = [100 + r, 101, 8, 48]
            rows.append(dict(uid="p{}".format(g), rep=r, prompt="[]", prompt_ids=[7, 50, 7, 8], response_ids=resp,
                             loss_mask=[1] * len(resp), score=float(10 * ((r + g) % 2)), stop_reason="stop"))
    if mutate:
        mutate(rows)
    path = str(tmp_path / "pregen.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _validate(path, n=4):
    return data.validate_pregen(path, n, max_seq_len=64, vocab_size=1000, eos_id=48, assistant_id=8)


def test_pregen_valid_file_has_no_errors(tmp_path):
    errors, warnings, stats = _validate(_parquet(tmp_path))
    assert errors == [] and warnings == []
    assert stats["rows"] == 24 and stats["prompts"] == 6 and stats["longest"] == 8


@pytest.mark.parametrize("mutate, expected", [
    (lambda rows: rows.pop(), "without exactly 4 rollouts"),
    (lambda rows: rows[0].update(loss_mask=[1, 1]), "len(loss_mask) != len(response_ids)"),
    (lambda rows: rows[0].update(loss_mask=[0, 0, 0, 0]), "no loss token"),
    (lambda rows: rows[0].update(loss_mask=[1, 1, 1, 0]), "loss_mask 0 on the final eos"),
    (lambda rows: rows[0].update(response_ids=[100, 101, 8, 5000]), "outside the vocabulary"),
    (lambda rows: rows[0].update(response_ids=[100, 101, 8, 9]), "do not end with eos"),
    (lambda rows: rows[0].update(prompt_ids=[7, 50, 7, 7]), "generation-prompt token"),
    (lambda rows: rows[0].update(response_ids=list(range(100, 170)), loss_mask=[1] * 70), "above max_seq_len"),
    (lambda rows: rows[0].update(score=11.0), "score outside"),
    (lambda rows: rows[1].update(rep=0), "duplicate (uid, rep)"),
    (lambda rows: [r.update(score=5.0) for r in rows], "no advantage"),
])
def test_pregen_violations_are_reported(tmp_path, mutate, expected):
    errors, _, _ = _validate(_parquet(tmp_path, mutate))
    assert any(expected in e for e in errors), errors


def test_pregen_lopsided_advantages_warn(tmp_path):
    def long_bad_rollouts(rows):  # every low-score rollout is 10x longer: the token-weighted gradient mostly pushes down
        for r in rows:
            if r["score"] == 0:
                r["response_ids"] = [100] * 38 + [8, 48]
                r["loss_mask"] = [1] * 40
    errors, warnings, stats = _validate(_parquet(tmp_path, long_bad_rollouts))
    assert errors == [] and stats["advantage_token_imbalance"] < -0.5
    assert any("lopsided" in w for w in warnings)


# ---------------------------------------------------------------- run log
def _log(tmp_path, steps, extra=""):
    lines = []
    for s in steps:
        d = dict({"trainer/global_step": s.pop("step")}, **s)
        body = ",\n".join("0: (skyrl_entrypoint pid=1)  '{}': {}".format(k, v if k == "trainer/global_step" else "'{:.4f}'".format(v))
                          for k, v in sorted(d.items()))
        lines.append(body + "}")
    p = tmp_path / "slurm-1.out"
    p.write_text("\n".join(lines) + "\n" + extra)
    return Ctx("/nonexistent", run_dir=str(tmp_path), log=str(p))


def _pregen_step(step, diff, clip, kl=0.001):
    return {"step": step, "generate/pregenerated_rows": 40, runlog.ABS_DIFF: diff, runlog.CLIP: clip, runlog.KL: kl}


def test_noise_floor_passes_at_measured_healthy_values(tmp_path):
    ctx = _log(tmp_path, [_pregen_step(1, 0.011, 0.0065), _pregen_step(2, 0.012, 0.007)])
    assert runlog.step1_noise_floor(ctx).status == PASS
    assert runlog.rollout_mismatch(ctx).status == SKIP
    assert runlog.run_errors(ctx).status == PASS


def test_noise_floor_fails_when_policy_and_reference_disagree_at_step_1(tmp_path):
    assert runlog.step1_noise_floor(_log(tmp_path, [_pregen_step(1, 0.24, 0.32)])).status == FAIL


def test_noise_floor_is_skipped_for_a_resumed_log(tmp_path):
    assert runlog.step1_noise_floor(_log(tmp_path, [_pregen_step(109, 0.037, 0.057)])).status == SKIP


def test_online_mismatch_fails_at_the_values_of_the_rope_bug(tmp_path):
    ctx = _log(tmp_path, [{"step": i, runlog.ROLLOUT_DIFF: 0.117, runlog.ABS_DIFF: 0.115} for i in (1, 2, 3)])
    assert runlog.rollout_mismatch(ctx).status == FAIL
    assert runlog.step1_noise_floor(ctx).status == SKIP


def test_online_mismatch_passes_at_qwen_values(tmp_path):
    ctx = _log(tmp_path, [{"step": 1, runlog.ROLLOUT_DIFF: 0.0105, runlog.ABS_DIFF: 0.0098}])
    assert runlog.rollout_mismatch(ctx).status == PASS


def test_errors_and_nan_fail(tmp_path):
    ctx = _log(tmp_path, [_pregen_step(1, float("nan"), 0.01)])
    assert runlog.run_errors(ctx).status == FAIL
    (tmp_path / "b").mkdir()
    ctx = _log(tmp_path / "b", [_pregen_step(1, 0.01, 0.01)], extra="slurmstepd: error: *** JOB 1 CANCELLED AT x DUE TO TIME LIMIT ***\n")
    assert runlog.run_errors(ctx).status == FAIL


def test_drift_warns_when_most_tokens_leave_the_band(tmp_path):
    steps = [_pregen_step(i, 0.01 + 0.002 * i, 0.01 + 0.004 * i, kl=0.001 * i) for i in range(1, 61)]
    assert runlog.drift(_log(tmp_path, steps)).status == WARN


# ---------------------------------------------------------------- model json keys, run settings
def _model_dir(tmp_path, cfg):
    import json
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return Ctx("/nonexistent", model_dir=str(tmp_path), cfg={})


def test_config_keys_flags_the_tildeopen_config(tmp_path):
    res = model.config_keys(_model_dir(tmp_path, {"rope_scaling": {"rope_type": "yarn", "factor": 10.0, "attention_factor": 1.0},
                                                  "apply_yarn_scaling": False}))
    assert res.status == WARN and len(res.details) == 2


def test_config_keys_passes_a_plain_yarn_config(tmp_path):
    assert model.config_keys(_model_dir(tmp_path, {"rope_scaling": {"rope_type": "yarn", "factor": 10.0}})).status == PASS


def test_ref_temperature():
    def ctx(temp, kl):
        return Ctx("/nonexistent", cfg={"skyrl": {"generator": {"sampling_params": {"temperature": temp}},
                                                 "trainer": {"algorithm": {"use_kl_loss": kl}}}})
    assert settings.ref_temperature(ctx(1.0, True)).status == PASS
    assert settings.ref_temperature(ctx(0.7, True)).status == FAIL
    assert settings.ref_temperature(ctx(0.7, False)).status == SKIP


def test_export_dtype_warns_for_the_runs_that_lost_their_update(tmp_path):
    rng = np.random.default_rng(0)
    write_safetensors(str(tmp_path / "model.safetensors"), {"model.layers.0.mlp.up_proj.weight": (rng.normal(0, 0.016, 20000), "BF16")})

    def ctx(lr, steps):
        return Ctx("/nonexistent", model_dir=str(tmp_path),
                   cfg={"skyrl": {"trainer": {"max_training_steps": steps, "policy": {"optimizer_config": {"lr": lr}}}}})
    assert settings.export_dtype(ctx(1e-6, 162)).status == WARN
    assert settings.export_dtype(ctx(1e-6, 1296)).status == WARN
    assert settings.export_dtype(ctx(2e-5, 1296)).status == PASS
    assert settings.export_dtype(ctx(1e-6, None)).status == SKIP


# ---------------------------------------------------------------- gpu parity verdict
def test_gpu_parity_verdict_thresholds():
    from checks.gpu_parity import verdict

    rng = np.random.default_rng(0)
    lp = -rng.exponential(1.0, 20000)
    noise = rng.standard_t(3, lp.size) * 0.02                           # bf16-like noise, heavy tail, no offset
    assert verdict(lp + noise, lp)["status"] == "PASS"
    assert verdict(lp + noise + 0.107, lp)["status"] == "FAIL"           # the systematic offset of the RoPE bug
    assert verdict(lp + rng.normal(0, 0.3, lp.size), lp)["status"] == "FAIL"   # no offset, but far too noisy
