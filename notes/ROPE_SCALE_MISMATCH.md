# Why the FSDP trainer and vLLM use different RoPE scales for TildeOpen

Written 2026-11-11. Versions: transformers 5.8.0, vLLM 0.22.1 (the ones in `multitorch_sandbox.sif`).

## Short version

- "FSDP" does not pick a scale. SkyRL's FSDP workers build the model with the **transformers**
  library; vLLM has its **own, separate implementation** of the same architecture. Both read the same
  `config.json`.
- YaRN has one extra number, a magnitude factor for the rotary embedding. Its standard value for
  this model is `0.1 * ln(10) + 1 = 1.2303`.
- The two libraries use **different names** for the config key that overrides this number.
  The config uses the transformers name (`attention_factor: 1.0`). transformers obeys it and uses 1.0.
  vLLM does not know that name, ignores it, and uses the standard 1.2303.
- Two tests say the checkpoint was trained with 1.2303. So the key in the config is wrong, vLLM has
  been right by accident, and every transformers-based run (the FSDP trainer, the ref worker) has
  been running a degraded network.
- Fix: delete `attention_factor` from `rope_scaling`. Then neither library is told anything, and
  both fall back to the same standard value.

## 1. What the number is

RoPE encodes a token's position by rotating its query and key vectors; the rotation is a table of
`cos` and `sin` values per position. YaRN (the method that stretched this model from 8k to 64k
context, `factor: 10`) changes two things:

1. **the rotation frequencies**, so that positions beyond 8k still look familiar. Both libraries
   compute these identically (`checks/rope_parity.py`: tables equal to 1e-7 once the scale is
   divided out);
2. **a magnitude factor** `s` that the `cos`/`sin` tables are multiplied with. The YaRN paper's
   recommended value is `s = 0.1 * ln(factor) + 1`. For `factor = 10` that is **1.2303**.

Queries and keys are both multiplied by `s`, so every attention score `q.k` is multiplied by
`s^2 = 1.51`. That is the same as running every attention layer at a lower temperature: the model
attends more sharply. A model trained with 1.51x sharper attention and run without it (or the other
way round) is a different function of the same weights, at every position, in all 60 layers.

## 2. How each library decides `s`

The model's `config.json`:

    "rope_scaling": {
        "rope_type": "yarn", "factor": 10.0,
        "attention_factor": 1.0,            <- the key in question
        "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 8192
    }

**transformers** (`modeling_rope_utils.py`, `_compute_yarn_parameters`):

    attention_factor = rope_parameters_dict.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0      # the YaRN default
    ...
    cos = emb.cos() * self.attention_scaling                  # modeling_llama.py
    sin = emb.sin() * self.attention_scaling

The key is present with value 1.0, so the default is skipped: **s = 1.0**.

**vLLM** (`model_executor/layers/rotary_embedding/__init__.py` and `yarn_scaling_rope.py`):

    extra_kwargs = {k: v for k, v in rope_parameters.items()
                    if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                             "beta_slow", "apply_yarn_scaling", "truncate")}
    ...
    self.mscale = yarn_get_mscale(scaling_factor) * attn_factor  if apply_yarn_scaling  else attn_factor

vLLM only forwards the keys in that list. `attention_factor` is not in it (vLLM's name is
`attn_factor`), so it is silently dropped; `attn_factor` stays at its default 1 and
`apply_yarn_scaling` at its default True: **s = 1.2303 * 1 = 1.2303**.

Side by side:

| | transformers (FSDP policy + ref workers) | vLLM (sampling, weight-synced engines) |
|---|---|---|
| key it reads to override `s` | `attention_factor` | `attn_factor`, `apply_yarn_scaling` |
| sees `attention_factor: 1.0`? | yes, obeys it | no, not in its list of keys |
| value used | **1.0** | **1.2303** |
| attention scores relative to the other | 1.0 | 1.51x |

Neither library has a bug in the usual sense. They disagree on the name of an optional key, and the
config happens to set that key under one of the two names. Most YaRN models on the hub do not set it
at all, which is why this rarely shows up (and why Qwen, which has no YaRN, never showed a gap).

The top-level `"apply_yarn_scaling": false` that was added to `tildeopen/config.json` on Oct 22 does
nothing: vLLM looks for that key inside the rope dict, not at the top level.

## 3. Which value is right

The config says 1.0, but two measurements say the weights were trained with 1.2303:

| test | scale 1.0 | scale 1.2303 |
|---|---|---|
| mean log-prob of dataset prompt tokens, 1,000 prompts, both in transformers (job 23641346) | -2.366 | **-2.228** |
| share of prompts where this scale fits better | 2.2% | **97.8%** |

The prompts are dataset text, not samples from either network, so the better fit is not an artefact
of who generated them. A gap of 0.14 nats per token in favour of 1.2303, on 98% of prompts and in
every length bucket, is far outside noise.

Not known: why the config contains `attention_factor: 1.0`. Most likely it was written when the
checkpoint was exported, by a tool or by hand, without that value having been used in training.
Confirming with the SFT owner is still worth it; the measurement is what the fix is based on.

## 4. What it did to the runs

| component | library | scale | consequence |
|---|---|---|---|
| vLLM that sampled the stored rollouts | vLLM | 1.2303 (right) | data is fine |
| vLLM engines in online runs (gsm8k, Harbor) | vLLM | 1.2303 (right) | sampling is fine |
| FSDP policy worker (the network being trained) | transformers | 1.0 (wrong) | gradients computed on a degraded network |
| FSDP ref worker (KL anchor, ratio reference) | transformers | 1.0 (wrong) | anchor is the degraded network |
| evaluation with vLLM | vLLM | 1.2303 (right) | trained weights judged on a network they were not optimised for |

- Per-token log-probs of the two networks differ by 0.24 nats on average (job 23611214); with the
  scales made equal the difference drops to 0.027.
- In training this meant: the network being optimised was 0.14 nats/token worse than the real model,
  and the data came from the real model. Part of every update was the weights compensating for the
  wrong scale. Served again at the right scale, that compensation is not what you want.
- Not shown directly: that this is what lowered the evaluation score. A rerun with the fixed config
  is the test.

## 5. The fix, and why this one

Make both libraries arrive at the same value, and make it the trained one (1.2303):

    "rope_scaling": {
        "rope_type": "yarn", "factor": 10.0,
        "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 8192
    }

- **Delete `attention_factor`.** transformers then takes its default branch, `0.1 * ln(10) + 1`,
  the same formula vLLM already uses. `checks/rope_parity.py` on this variant: both x1.2303, tables
  identical.
- **Delete the top-level `apply_yarn_scaling`.** It has no effect; it only misleads.
- **Do not add `apply_yarn_scaling: false` inside the rope dict.** That was my suggestion before the
  scale test. It also makes the libraries agree, but at 1.0: it would switch vLLM to the wrong
  network instead of switching transformers to the right one.

Files that carry the key and need the change before transformers loads them:
- `skyrl-training/tildeopen/config.json` (hard-linked with `fsdp-skyrl-harbor/tildeopen/config.json`:
  one edit changes both)
- `grpo-pregen-rl/tildeopen/config.json`
- `config.json` inside every HF export (there the dict is called `rope_parameters`)

After editing: `checks/rope_parity.py <model dir>` must print `SAME` at x1.2303, and a gsm8k mismatch
run should show `policy/minibatch_rollout_logprobs_abs_diff_mean` around 0.01-0.03 instead of 0.10-0.21.
Existing checkpoints from the pregenerated runs were trained against the scale-1.0 network and
should be retrained, not reused.
