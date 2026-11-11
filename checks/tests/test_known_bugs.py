"""
Negative controls: every bug this suite was written for must still be caught, and the repaired
state must pass. Needs the container (transformers + vLLM), a TildeOpen model dir (its golden file is
checks/golden/tildeopen.jsonl), and for the last test one of the old exports. LUMI only:

    ./check.sh selftest                # everything, ~4 min
    ./check.sh selftest -k "not export"   # without the slow export test

Each test builds its own copy of the model dir (weights symlinked) in a temp dir, so the result
does not depend on whether the real model dir has been repaired yet.
"""
import os

import pytest

from checks import export, model, tokenizer
from checks.lib import FAIL, PASS, WARN, Ctx
from checks.tests.conftest import BAD_EXPORT, EXPORT_BASE, MODEL_DIR

pytest.importorskip("vllm")
pytest.importorskip("transformers")
needs_model = pytest.mark.skipif(not (os.path.isdir(MODEL_DIR) and os.path.exists(tokenizer.golden_path(MODEL_DIR))),
                                 reason="no model dir at {} or no golden file for it".format(MODEL_DIR))


def rope(c):
    return c["rope_parameters"] if "rope_parameters" in c else c["rope_scaling"]


def broken_rope(c):      # the config as it was shipped
    rope(c)["attention_factor"] = 1.0
    c["apply_yarn_scaling"] = False


def fixed_rope(c):       # neither library is told a magnitude: both use the YaRN default
    rope(c).pop("attention_factor", None)
    c.pop("apply_yarn_scaling", None)


def ctx(model_dir):
    return Ctx("/nonexistent", model_dir=model_dir, cfg={}, opts={})


@needs_model
def test_rope_bug_is_caught(make_variant):
    d = make_variant("broken", {"config.json": broken_rope})
    res = model.rope_parity(ctx(d))
    assert res.status == FAIL
    assert res.data["transformers_scale"] == pytest.approx(1.0) and res.data["vllm_scale"] == pytest.approx(1.2303, abs=1e-4)
    assert model.config_keys(ctx(d)).status == WARN


@needs_model
def test_rope_fix_passes(make_variant):
    d = make_variant("fixed", {"config.json": fixed_rope})
    res = model.rope_parity(ctx(d))
    assert res.status == PASS and res.data["transformers_scale"] == pytest.approx(res.data["vllm_scale"])
    assert model.config_keys(ctx(d)).status == PASS


@needs_model
def test_rope_wrong_fix_is_still_flagged(make_variant):
    """apply_yarn_scaling=false inside the rope dict makes the tables equal, at the wrong magnitude (1.0)."""
    def wrong(c):
        broken_rope(c)
        rope(c)["apply_yarn_scaling"] = False
    d = make_variant("wrong", {"config.json": wrong})
    assert model.rope_parity(ctx(d)).status == PASS            # the libraries agree ...
    assert model.config_keys(ctx(d)).status == WARN            # ... but the one-sided keys are still reported


@needs_model
def test_tokenizer_bug_is_caught(make_variant):
    d = make_variant("tok_broken", {"tokenizer_config.json": lambda t: t.update(add_prefix_space=False)})
    res = tokenizer.tokenizer_golden(ctx(d))
    assert res.status == FAIL and res.data["identical"] == 0


@needs_model
def test_tokenizer_fix_passes(make_variant):
    d = make_variant("tok_fixed", {"tokenizer_config.json": lambda t: t.update(add_prefix_space=True)})
    res = tokenizer.tokenizer_golden(ctx(d))
    assert res.status == PASS and res.data["identical"] == res.data["total"] > 0


@needs_model
def test_missing_golden_file_is_a_warning_not_a_pass(make_variant):
    d = make_variant("no_golden", {}, dir_name="a_model_without_a_golden_file")
    if os.path.exists(os.path.join(d, tokenizer.GOLDEN)):     # an override copied along from the real model dir
        os.remove(os.path.join(d, tokenizer.GOLDEN))
    assert tokenizer.tokenizer_golden(ctx(d)).status == WARN


@needs_model
def test_golden_file_inside_the_model_dir_overrides_the_repo_one(make_variant):
    d = make_variant("override", {"tokenizer_config.json": lambda t: t.update(add_prefix_space=True)})
    with open(os.path.join(d, tokenizer.GOLDEN), "w") as f:
        f.write('{"messages": [{"role": "user", "content": "hi"}], "ids": [1, 2, 3]}\n')
    assert tokenizer.golden_path(d) == os.path.join(d, tokenizer.GOLDEN)
    assert tokenizer.tokenizer_golden(ctx(d)).status == FAIL       # the (deliberately wrong) override is what gets used


@pytest.mark.skipif(not os.path.isdir(os.path.join(BAD_EXPORT, "policy")), reason="old export not available")
def test_export_that_lost_its_update_is_caught():
    """exports/replay/global_step_162: 162 steps at lr 1e-6, measured cosine 0.52 over all tensors."""
    c = Ctx("/nonexistent", export_dir=BAD_EXPORT, base_dir=EXPORT_BASE, opts={})
    old, export.MAX_TENSORS = export.MAX_TENSORS, 24
    try:
        res = export.export_delta(c)
    finally:
        export.MAX_TENSORS = old
    assert res.status == FAIL and 0.4 < res.data["ALL"]["cos"] < 0.65
    assert any(d.endswith("global_step_162/policy") for d in res.details)
