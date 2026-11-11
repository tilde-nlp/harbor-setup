# Sanity check of the pregenerated-rollout GRPO run (TildeOpen-30B)

Written 2026-11-10. Covers the two runs in `grpo-pregen-rl` (bs40 x 162 steps, bs5 x 1296 steps), the
installed SkyRL / transformers 5.8.0 / vLLM 0.22.1, and the job logs in `fsdp-skyrl-harbor`.
Scripts that reproduce every number: `checks/` (see `checks/README.md`).

Five problems, most serious first. None of them is an hparam issue. Which one (if any) made the
evaluated model worse is not known yet; that depends on how the evaluation was run (section 6).

| # | problem | status |
|---|---|---|
| 1 | vLLM and the trainer compute different models from the same weights (RoPE scale 1.23 vs 1.0) | confirmed by `checks/rope_parity.py` |
| 2 | Trainer and vLLM log-probs disagree for TildeOpen, and SP is not the reason | confirmed from logs |
| 3 | The bf16 model that gets evaluated contains only part of the trained update | measured by `checks/export_delta.py` |
| 4 | The tokenizer in this container does not produce the token ids the model was trained and sampled with | confirmed by `checks/tokenizer_parity.py` |
| 5 | The importance ratio in the loss is measured against the wrong reference and is noisy | confirmed from logs and code |

---

## 1. vLLM and the trainer disagree on the RoPE scale

`tildeopen/config.json` says:

    "rope_scaling": {"rope_type": "yarn", "factor": 10.0, "attention_factor": 1.0, ...}

YaRN multiplies the rotary cos/sin tables by a magnitude factor. The two libraries pick it differently:

| library | where it runs | reads | factor used |
|---|---|---|---|
| transformers | FSDP policy and ref workers | `attention_factor` | **1.0** |
| vLLM | sampling, weight-synced engines, any vLLM evaluation | `attn_factor`, `apply_yarn_scaling` (does not know `attention_factor`) | **0.1 * ln(10) + 1 = 1.2303** |

Result of `checks/rope_parity.py`:

    config                                              transformers   vLLM
    as is (grpo-pregen-rl/tildeopen, all exports)       x1.0000        x1.2303   DIFFERENT
    as is (skyrl-training/tildeopen, top-level key)     x1.0000        x1.2303   DIFFERENT
    rope_scaling.apply_yarn_scaling=false               x1.0000        x1.0000   SAME
    rope_scaling without attention_factor               x1.2303        x1.2303   SAME

- The factor scales both query and key rotations, so attention logits differ by about 1.23^2 = 1.51
  in every layer. Same weights, two different networks.
- `"apply_yarn_scaling": false` in `skyrl-training/tildeopen/config.json` (and `fsdp-skyrl-harbor/tildeopen`)
  is at the top level of the file. vLLM only looks inside the rope dict, so the key does nothing.
- Two configs make the libraries agree: put `apply_yarn_scaling: false` inside `rope_scaling` (both use 1.0)
  or drop `attention_factor` (both use 1.23). **Which is right depends on what the SFT training used.
  This is unknown and has to be asked** (section 6). The wrong choice makes both libraries
  consistently wrong.
- Not proven yet: that this accounts for the whole log-prob gap of section 2.
  `checks/logprob_parity.sbatch` (one node) tests it.

## 2. Trainer and vLLM log-probs disagree, and it is not SP

`policy/minibatch_rollout_logprobs_abs_diff_mean` (mean |log p_trainer - log p_vLLM| per sampled token)
in the first steps of past jobs:

| model | SP | abs diff |
|---|---|---|
| Qwen3-14B | 1 and 4 | 0.008 - 0.011 |
| TildeOpen | 4 | 0.10 - 0.21 |
| TildeOpen | 1 (jobs 23422469, 23422900) | 0.115, 0.102 |

The gap follows the model, not SP. The one "TildeOpen, SP=1, 0.0098" job (22587370) ran
`MODEL_SRC=Qwen/Qwen3-14B` despite its job name. Section 1 is the likely cause.

## 3. The evaluated bf16 model contains only part of the trained update

**What happens.** The base checkpoint is stored in bf16. The trainer loads it into fp32 and trains in
fp32. The export is fp32 (120 GB), and `hf_to_bf16.py` then casts it to bf16 (60 GB) for evaluation.

**Why that loses the update.** bf16 has 8 bits of mantissa. Around a typical weight of 0.016 the
representable values are 0.000122 apart:

    ... 0.015991   0.016113   0.016235 ...      (bf16 grid, step 1.2e-4)

Every base weight sits exactly on a grid point. Training at lr 1e-6 moves a weight by about 1e-5 in
162 steps, a tenth of one grid step. Casting back to bf16 rounds to the nearest grid point:

    base 0.016113  -> trained 0.016124 (moved +1.1e-5)  -> bf16 0.016113   update erased
    base 0.016113  -> trained 0.016180 (moved +6.7e-5)  -> bf16 0.016235   update doubled (+1.2e-4)

So a weight either snaps back to its base value or jumps a full grid step. Small updates vanish, the
few large ones are exaggerated.

**Measured** (`checks/export_delta.py`, all 543 tensors, first 200k elements of each):

| run | weights that changed at all in bf16 | cosine(true update, exported update) | norm weights changed |
|---|---|---|---|
| bs40, step 162 | 14.7% | 0.52 | 0.8% |
| bs5, step 1296 | 35.5% | 0.77 | 1.1% |

- Cosine 1.0 would mean the bf16 model carries the trained update. At 0.52, roughly a quarter of the
  change in the bf16 file is the update and three quarters is rounding error; at 0.77 it is 59%.
- RMSNorm weights (around 0.8-1.7, grid step 0.004-0.008) lose their update almost completely.
- Not established: that the rounding error by itself lowers quality (its size is about 5e-4 of the
  weight norm). What is established: the evaluation did not measure the model that was trained.
- The same cast happens every step in online runs (`model_dtype: bfloat16` for the weight sync), so the
  vLLM engines sample from a policy that lags the trainer's.

**What to do.** Evaluate the fp32 export (`.../policy`, vLLM `--dtype float32` if memory allows), or
train long/hard enough that the update is well above the grid and check with `export_delta.py`
before evaluating. A larger learning rate is the usual reason other setups do not see this.

## 4. The tokenizer in this container does not reproduce the training tokenization

**What the model expects.** In the stored rollouts (and per the data README, in everything the model
saw) the first word after a control token starts with the word-boundary marker `▁`:

    <|user|> ▁Pro vide ▁a ▁clear ...          ids 7, 1210, 30662, 460, 17109

**What this container produces.** transformers 5.8.0 loads `tokenizer.model` as a converted fast
tokenizer. With `"add_prefix_space": false` in `tokenizer_config.json` it does not add that marker:

    <|user|> Prov ide ▁a ▁clear ...            ids 7, 23360, 1052, 460, 17109

**Measured** (`checks/tokenizer_parity.py`, 300 prompts rendered with the chat template):

| tokenizer directory | prompts identical to the stored ids |
|---|---|
| `tildeopen` (base, as loaded by SkyRL and vLLM here) | 0 / 300 |
| `grpo-pregen-rl/tildeopen` | 0 / 300 |
| export `policy_bf16` (has a converted `tokenizer.json`) | 0 / 300 |
| `tildeopen` loaded with `add_prefix_space=True` | **300 / 300** |
| export loaded with `add_prefix_space=True` | 0 / 300 (the setting is baked into its `tokenizer.json`) |

Every prompt differs, always at the first token after `<|user|>` or `<|system|>`. Plain sentencepiece
on `tokenizer.model` gives the stored ids, so the data is right and the loading is wrong.

**What it affects.**
- *Pregenerated training itself: not affected.* It feeds stored token ids; no tokenizer is involved.
- *Evaluation of an export:* the export ships the wrong `tokenizer.json`. Any engine that loads the
  tokenizer from the export directory prompts the model with token sequences it never saw at the start
  of each turn. If the base model was evaluated in an environment that tokenizes correctly (older
  transformers, sentencepiece) and the trained one from its export directory, the comparison is unfair
  to the trained model regardless of training. Whether that happened is unknown (section 6).
- *Online runs in this container (gsm8k, Harbor):* vLLM tokenizes prompts and Harbor/SkyRL render chat
  turns with this tokenizer, so TildeOpen has been sampled on slightly out-of-distribution prompts in
  every online run. Not checked: whether tool-loop text inserted mid-response has the same issue.

**What to do.** Set `"add_prefix_space": true` in the base `tokenizer_config.json`, re-run
`tokenizer_parity.py` (must be 300/300), and for existing exports replace the tokenizer files with the
base ones instead of the generated `tokenizer.json`. Whoever evaluates should run the same check in
their environment.

## 5. The importance ratio is measured against the wrong reference, and it is noisy

**What the loss does.** For every token of a stored rollout:

    ratio  = p_policy_now(token) / p_reference(token)
    weight = ratio if 0.8 < ratio < 1.2 else 0          (the "band", eps_clip 0.2)
    loss   = - weight * advantage * log p_policy_now(token)

The ratio is meant to answer "how far has the policy moved from the policy that *sampled* this
token?". Tokens where it moved more than 20% get no gradient. `p_reference` comes from SkyRL's ref
worker: the frozen start checkpoint, run through transformers every step.

**Problem A: the reference is not the policy that sampled the data.** The rollouts were sampled by
vLLM. Per sections 1-2, vLLM's distribution for this checkpoint is 0.1-0.2 nats per token away from
what transformers computes (per-token probability ratios of up to 9-20x were logged in job 23421119).
So:
- the ratio is 1 at step 1 by construction, which says "perfectly on-policy", while the data is in
  fact off-policy from the first step;
- the band limits the distance to the transformers-view of the start checkpoint, not to the data;
- the KL term pulls toward the same transformers-view.
The training signal is still "raise log-prob of above-average rollouts, lower it for below-average
ones", so this is a bias, not a sign error. It becomes harmless once section 1 is fixed and both
libraries compute the same model.

**Problem B: the ratio is noisy even when nothing has changed.** At step 1 the policy and the
reference have identical weights, so every ratio should be exactly 1. Logged instead:

| run | mean abs log-prob diff | largest diff | largest ratio | tokens already outside the band |
|---|---|---|---|---|
| bs40, step 1 | 0.011 | 1.73 nats | 2.7 | 0.65% |
| bs5, step 1 | 0.024 | - | - | 1.2% |

- A token can come out 2.7x more likely in one forward pass than in the other with the same weights.
- The two passes differ in: `model.train()` vs `eval()`, DP=8 vs DP=4 (different packing of sequences
  into micro-batches), and bf16 autocast rounding. Which of these dominates was not isolated.
- Consequence: around 1% of tokens are dropped from the gradient at random, and late in training part
  of what is read as "policy drift" (out-of-band 5-8% for bs40, about 20% for bs5) is this noise.
  The 0.01 floor is also the size of the Qwen train/inference gap, so it is normal for bf16 at this
  depth; it is a limit on how finely the band can discriminate, not a bug in the custom code.
- Small related defect: the ref worker's forward (`worker.py`, `RefWorkerBase._forward_micro_batch`)
  does not pass `temperature`. No effect at 1.0; wrong ratios at any other value.

**What to watch.** `policy/minibatch_rollout_logprobs_abs_diff_mean` and `policy/loss_metrics/clip_ratio`
at step 1 are the noise floor of a run. Compare later values against those, not against zero.

---

## 6. Open questions that decide the ranking

1. **How was the evaluation run?** Engine, which directory (`policy` fp32 or `policy_bf16`), and which
   tokenizer files for the base and for the trained model. Decides whether 3 and 4 explain the drop.
2. **Which RoPE magnitude did the SFT training use?** 1.0 (then vLLM is wrong today, including when it
   sampled the stored rollouts) or 1.23 (then the trainer is wrong). Decides the fix for 1.

## 7. Checked and found sound

- Stored data (6,400-row sample): loss mask is 0 only on real tool outputs (0.5% of tokens); every
  finished rollout ends with `<|assistant|><|endoftext|>`; positive and negative advantages carry
  about the same token mass; rollout length does not depend on score.
- `prepare_replay_data.py`, the (uid, rep) lookup, right-aligned response tensors, GRPO grouping,
  `token_mean` scaling, the `replay_offline` loss (same form as built-in `rollout_is`), KL wiring.
- Ulysses SP with sample packing: position ids are gathered across SP ranks before flash-attention
  derives sequence boundaries, so packed sequences do not attend to each other.
- FSDP shard merge: the bf16 export equals round(fp32 export) on every sampled value.
- Optimisation was calm: grad norm 0.15-0.45, policy KL 0.0007 -> 0.005 (bs40).

---

## 8. Update 2026-11-11: results of the log-prob parity job (23611214)

`checks/logprob_parity.sbatch`: 200 stored rollouts, 159,158 response tokens, vLLM vs plain
transformers on one node (no FSDP, no SP).

| pair | mean abs diff | median | p99 | mean (first - second) |
|---|---|---|---|---|
| vLLM as is vs transformers | 0.243 | 0.073 | 1.90 | +0.107 |
| vLLM with `apply_yarn_scaling=false` inside the rope dict vs transformers | 0.027 | 0.004 | 0.22 | +0.001 |

- **Section 1 is the cause of section 2.** The config change removes about 90% of the gap and all of
  the systematic offset. The remaining 0.027 is tail-dominated (median 0.004).
- **Rollout length does not matter** (`checks/logprob_gap_breakdown.py`): the gap is 0.17 in the first
  16 response tokens, 0.23 at tokens 64-256, flat at 0.25 from token 256 on. Rollouts under 300 tokens:
  0.21; over 2000 tokens: 0.25. The factor acts on every attention score; it is not the long-context
  part of YaRN.
- **What it means for section 5 (the pregenerated run).**
  - The stored tokens are on average 0.107 nats more likely under the network that sampled them
    (vLLM, scale 1.23) than under the trainer's network (scale 1.0). Training moved the policy 0.005
    (bs40) to 0.03 (bs5) nats from the reference, so the reference was 4 to 20 times further from the
    data than the policy ever got from the reference.
  - Over a whole rollout this adds up: for a 600-1000-token rollout the sampler's probability is a
    median e^74 times the trainer network's.
  - 32% of tokens have a true ratio p_trainer/p_sampler outside the (0.8, 1.2) band before any
    training. The loss saw a ratio of 1 for all of them. With the RoPE fix the figure is 1.4%.
- **Still open:** which scale the SFT checkpoint was trained with. Ask the owner what the training
  code did (not "which attention_factor": that is a transformers key). Empirical test, one node,
  not run yet: `sbatch checks/rope_scale_test.sbatch` scores the dataset prompts with transformers
  under both scales; the trained scale should give the higher mean log-prob.
