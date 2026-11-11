"""
Checks on the model directory: do transformers (FSDP policy and ref workers) and vLLM (sampling,
weight-synced engines, most evaluations) build the same network from its config.json?

Background: notes/ROPE_SCALE_MISMATCH.md. TildeOpen's config set `attention_factor: 1.0`, a key
transformers obeys and vLLM does not know: the trainer ran RoPE magnitude 1.0, vLLM 1.2303.
"""
import copy
import json
import os

from checks.lib import FAIL, PASS, SKIP, WARN, Result, check

TOLERANCE = 1e-4          # max |difference| between the two cos/sin tables; equal configs give < 1e-6
POSITIONS = 4096
# keys of the rope dict that only one library reads (transformers 5.8.0, vLLM 0.22.1)
TRANSFORMERS_ONLY = {"attention_factor", "mscale", "mscale_all_dim"}
VLLM_ONLY = {"attn_factor", "apply_yarn_scaling", "extrapolation_factor"}


def load_config(path):
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    return json.load(open(path))


def rope_key(cfg):
    return "rope_parameters" if "rope_parameters" in cfg else "rope_scaling"


def _hf_tables(cfg, n_pos):
    import importlib

    import torch
    from transformers import CONFIG_MAPPING

    model_type = cfg["model_type"]
    kwargs = {k: v for k, v in copy.deepcopy(cfg).items() if k not in ("architectures", "model_type", "transformers_version")}
    c = CONFIG_MAPPING[model_type](**kwargs)
    mod = importlib.import_module("transformers.models.{0}.modeling_{0}".format(model_type))
    cls = next(v for k, v in vars(mod).items() if k.endswith("RotaryEmbedding"))  # StopIteration: model has none
    rope = cls(c)
    cos, sin = rope(torch.zeros(1, n_pos, 1), torch.arange(n_pos)[None])
    half = cos.shape[-1] // 2
    return cos[0, :, :half].double(), sin[0, :, :half].double(), float(getattr(rope, "attention_scaling", 1.0)), dict(c.rope_parameters)


def _vllm_tables(rope_parameters, cfg, n_pos):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    head = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    if hasattr(get_rope, "cache_clear"):
        get_rope.cache_clear()
    # the layer is a vLLM CustomOp: it can only be built while a vLLM config is "current"; cpu = login node
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        # same call as vllm/model_executor/models/llama.py: rope_parameters comes from the HF config object
        rope = get_rope(head, max_position=cfg["max_position_embeddings"], rope_parameters=dict(rope_parameters), is_neox_style=True)
    cache = rope.cos_sin_cache[:n_pos].double()
    half = cache.shape[-1] // 2
    return cache[:, :half], cache[:, half:], float(getattr(rope, "mscale", 1.0))


def rope_tables_diff(cfg, n_pos=POSITIONS):
    """-> (max |diff| of cos/sin tables, transformers magnitude factor, vLLM magnitude factor)"""
    h_cos, h_sin, h_scale, rp = _hf_tables(cfg, n_pos)
    v_cos, v_sin, v_scale = _vllm_tables(rp, cfg, n_pos)
    d = max((h_cos - v_cos).abs().max().item(), (h_sin - v_sin).abs().max().item())
    return d, h_scale, v_scale


@check("pre", "model.rope_parity", cache_inputs=lambda ctx: [os.path.join(ctx.model_dir, "config.json")])
def rope_parity(ctx):
    """transformers and vLLM must build identical rotary (RoPE) tables from the model's config.json."""
    cfg = load_config(ctx.model_dir)
    if not ctx.opts.get("uses_vllm", True) and not ctx.opts.get("strict"):
        note = "  (this config starts no vLLM; still matters for whoever samples from or evaluates the result)"
    else:
        note = ""
    try:
        d, h, v = rope_tables_diff(cfg)
    except StopIteration:
        return Result(SKIP, "model type {} has no rotary embedding class in transformers".format(cfg.get("model_type")))
    data = {"max_abs_diff": d, "transformers_scale": h, "vllm_scale": v}
    if d <= TOLERANCE:
        return Result(PASS, "transformers x{:.4f}, vLLM x{:.4f}, tables equal".format(h, v), data=data)
    return Result(FAIL, "transformers and vLLM build different rotary tables: x{:.4f} vs x{:.4f}, max |diff| {:.2e}{}".format(h, v, d, note),
                  ["same weights, two different networks: per-token log-probs differed by 0.24 nats for TildeOpen",
                   "rope dict: {}".format(json.dumps(cfg.get(rope_key(cfg)))),
                   "see model.config_keys below and notes/ROPE_SCALE_MISMATCH.md"], data)


@check("pre", "model.config_keys")
def config_keys(ctx):
    """Rope keys in config.json that only one of the two libraries reads (the usual cause of a rope_parity FAIL)."""
    cfg = load_config(ctx.model_dir)
    problems = []
    rope = cfg.get(rope_key(cfg)) or {}
    for k in sorted(set(rope) & TRANSFORMERS_ONLY):
        problems.append("{}.{}={!r}: read by transformers, ignored by vLLM".format(rope_key(cfg), k, rope[k]))
    for k in sorted(set(rope) & VLLM_ONLY):
        problems.append("{}.{}={!r}: read by vLLM, ignored by transformers".format(rope_key(cfg), k, rope[k]))
    for k in sorted((TRANSFORMERS_ONLY | VLLM_ONLY) & set(cfg)):
        problems.append("top-level {}={!r}: neither library reads it at the top level".format(k, cfg[k]))
    if "rope_scaling" in cfg and "rope_parameters" in cfg:
        problems.append("both rope_scaling and rope_parameters are present")
    if not problems:
        return Result(PASS, "no rope key that only one library reads")
    return Result(WARN, "{} config key(s) that the two libraries treat differently".format(len(problems)), problems)
