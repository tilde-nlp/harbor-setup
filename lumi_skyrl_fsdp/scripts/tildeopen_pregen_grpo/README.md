# Pre-generated GRPO with SkyRL (replay package)

Trains `anonymous-17010-slow` with GRPO on rollouts that were already generated, tool-executed,
judged and tokenized. SkyRL's generator is replaced by one that returns the stored rollouts, so
no vLLM generation happens. Everything else (reference model, KL loss, GRPO advantages, FSDP
training) is stock SkyRL. Nothing inside the SkyRL checkout is modified.

Files:

| file | role |
|---|---|
| `prepare_replay_data.py` | converts the tokenized JSONL (one row per rollout) into `replay.parquet`, validates groups and masks |
| `generator.py` | `ReplayDataset` (one item per prompt), `ReplayStore` (parquet in memory), `ReplayGenerator` (returns stored rollouts by `(uuid, rollout_number)`) |
| `main_replay.py` | training entrypoint: `BasePPOExp` subclass with the replay generator and dataset |
| `submit_tildeopen_replay_4nodes.sbatch` | LUMI launch script, derived from the gsm8k one |
| `test_loss_masking.py` | CPU test: the installed SkyRL loss functions give zero loss and zero gradient on `loss_mask == 0` tokens |
| `inspect_dump.py` | checks a dumped training batch against the parquet: masks, response window, advantages |
| `verify_masks.py` | optional, needs the tokenizer: decodes mask-0 spans and checks they are exactly the real tool outputs |

Verified against SkyRL `main` as of 2026-10-26 (GeneratorInterface, GeneratorOutput fields,
BasePPOExp hooks, prepare_generator_input, preprocess layout, worker loss path). If the container
has an older SkyRL, step 1 below catches interface differences.

---

## Step 0. Install the package

```
cp -r replay/ $ROOT/replay/          # $ROOT = /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
cp submit_tildeopen_replay_4nodes.sbatch <where you keep sbatch files>/
```

`replay/` must contain `__init__.py`. The sbatch puts `$ROOT` on `PYTHONPATH`, so the package is
importable as `replay` inside the container.

## Step 1. Check the SkyRL interfaces in the container (2 minutes, CPU)

Inside the container (`singularity exec -B "$BINDS" "$SANDBOX" bash`, then `source` the sbatch
to get the environment, or `. /opt/venv/bin/activate` by hand), from `$ROOT`:

```
python -c "
from skyrl.train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.config import SkyRLTrainConfig
import inspect
print(GeneratorOutput.__annotations__.keys())
print(inspect.signature(BasePPOExp.get_generator))
print([k for k in SkyRLTrainConfig().trainer.__dataclass_fields__ if 'dump' in k])
"
python -m replay.test_loss_masking
```

Expect: `GeneratorOutput` has `prompt_token_ids, response_ids, rewards, loss_masks, stop_reasons,
rollout_metrics, rollout_logprobs, trajectory_ids` (more keys are fine); `get_generator(self, cfg,
tokenizer, inference_engine_client)`; `['dump_data_batch']`; and `ALL OK` from the loss test.
If `test_loss_masking.py` fails with a signature error, the error prints the installed function's
signature; adjust the keyword names in `call_with_known_args(...)` calls.

## Step 2. Build the parquet

```
python -m replay.prepare_replay_data \
    --jsonl /fast/TildeOpen/data/rl/prepared_jsonl/nvidia/Nemotron-RL-Science-v1/tokenized/so_openq.rollouts.train.tokens.jsonl \
    --out $DATA_DIR/replay.parquet
```

Reads: `uuid, rollout_number, prompt, prompt_token_ids, rollout_token_ids, rollout_loss_mask,
score, answer_true`. Checks per row (mask length, mask values, prompt `6/7 ... 8`, closing `[8,48]`
has mask 1) and per group (8 contiguous rows, `rollout_number` 0..7). Keeps a group only if some
rollout has `answer_true` and the 8 scores are not all equal (`--no-filter` to keep all).

Compare the printed `rollouts_out` and `tokens_loss` with `rollouts` and `loss_tokens` in
`so_openq.rollouts.train.tokens` (the stats file). Note the last line,
`max_prompt_plus_response`: `trainer.algorithm.max_seq_len` (= `MAX_MODEL_LEN` in the sbatch)
must be at least that. Rows longer than `--max-seq-len` (default 65536) are cut and marked `length`.

Optional, if you have the tokenizer: `python -m replay.verify_masks --parquet $DATA_DIR/replay.parquet --model <tokenizer dir> --examples 5`.

## Step 3. Smoke run (1 step, with a batch dump)

```
TRAIN_BS=8 MINI_BS=8 EXTRA_ARGS="trainer.dump_data_batch=true trainer.max_steps=1" \
    sbatch submit_tildeopen_replay_4nodes.sbatch
```

(`trainer.max_steps` may be named differently in your version; if it is rejected, drop it and let
the job run until its time limit or cancel it after step 1.) `EXTRA_ARGS` is not in the original
sbatch; add this line at the end of `role_train` instead of the plain call, or edit the python
line by hand:

```
python -m replay.main_replay "${ARGS[@]}" ${EXTRA_ARGS:-} "$@"
```

What to look for in the log:
* `ReplayStore: 52112 rollouts, 6514 prompts` and `ReplayDataset: 6514 prompts x 8 rollouts` (your numbers).
* `generate/avg_reward` or `avg_score`: a mean of your 0..10 scores, not 0/1.
* `generate/avg_assistant_tokens` is `loss_tokens / rollouts` from the stats file;
  `generate/avg_num_tokens` is larger by the masked tool-output tokens.
* `policy/kl_loss` about 0 and `policy/ppo_clip_ratio` about 0 at step 1 (policy == reference).
* No `rollout_logprobs` assertion: off-policy correction is off, the generator returns `None`.

Then check the dumped batch:

```
python -m replay.inspect_dump \
    --dump <trainer.export_path>/dumped_data/global_step_1_training_input.pkl \
    --parquet $DATA_DIR/replay.parquet
```

Expect `matched+ok 64   problems 0   unmatched 0   of 64` (TRAIN_BS * 8) and the rewards list
showing your scores.

## Step 4. Full run

Remove `trainer.dump_data_batch=true` and the step limit. Defaults in the sbatch:
`TRAIN_BS=8`, `MINI_BS=8`, `N_SAMPLES=8`, `trainer.epochs=1` (one pass over 6,514 prompts
is 814 steps at 8 prompts/step; raise `TRAIN_BS`/`MINI_BS` to 32 or 64 for fewer, larger steps,
keeping the batch rule `MINI_BS * 8 / DP` divisible by `MICRO_TRAIN_BS`), `lr=1e-6`,
`use_kl_loss=true`. Set `trainer.ckpt_interval` to something positive for a real run.

Watch `policy/kl_loss` and `policy/rollout_train_logprobs_abs_diff_*` (only logged when
rollout_logprobs exist, see step 5). The data is fixed, so as the policy drifts from the
checkpoint that wrote it, the updates become off-policy; `use_kl_loss` keeps that bounded.

## Step 5 (optional). Off-policy correction

TIS needs behaviour log-probs. The generator passes a `ref_logprobs` column through as
`rollout_logprobs` if the parquet has one (one float per response token, 0.0 on masked
positions is fine). Compute it once with vLLM `prompt_logprobs=1` on `prompt_ids + response_ids`
for the same checkpoint, add the column, then uncomment in the sbatch:

```
trainer.algorithm.off_policy_correction.tis_ratio_type="token"
trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0
```

## Layout and config notes

* `INFER_NODES=1`: SkyRL still launches vLLM engines and syncs weights to them each step; the
  replay generator never calls them. One node with `TP=4` (2 idle engines) is the minimum the
  sbatch's checks allow. `REF_NODES=1` holds the reference model for the KL loss; alternatively
  `REF_NODES=0` with `trainer.placement.colocate_policy_ref=true`.
* `generator.n_samples_per_prompt=8` must equal the group size in the parquet (asserted at startup).
* `generator.apply_overlong_filtering` stays `false`: `true` would zero the loss mask of every
  rollout that did not end with `[8,48]` (length and tool-call-limit rollouts).
* `generator.sampling_params.temperature=1.0` matches generation.
* Dataset shuffling: SkyRL shuffles prompts, not rollouts; the 8 rollouts of a prompt always
  travel together because they come from one dataset item.

## How masking works in SkyRL (verified in source)

`loss_masks` -> `convert_prompts_responses_to_batch_tensors` (right-aligned `(B, max_response)`
tensor, zeros on padding) -> `TrainingInputBatch["loss_mask"]` -> `PolicyWorkerBase._forward_backward_micro`:
`policy_loss_fn(..., loss_mask=loss_mask)` ending in `reduce_loss = (loss * loss_mask).sum()`;
`compute_approx_kl(..., loss_mask)` with `torch.where(loss_mask.bool(), kld * loss_mask, 0.0)`;
`masked_mean(entropy, loss_mask)`. GRPO advantages are multiplied by `response_mask` (all response
tokens), not `loss_mask`; that is why `inspect_dump.py` expects non-zero advantages on tool-output
tokens and `test_loss_masking.py` shows they produce no gradient. Masked tokens remain in the
context, as at generation time.