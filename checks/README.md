# checks: the "unit tests" of this RL setup

Each check answers one question whose wrong answer has already cost a run (or would): does the
trainer compute the same network as vLLM, does the tokenizer produce what the model was trained on,
is the data well-formed, does the exported model contain the update. Why each exists:
`notes/TRAINING_SANITY_CHECK.md`, `notes/ROPE_SCALE_MISMATCH.md`.

## When to run what

| command | when | time | what a FAIL means |
|---|---|---|---|
| `./check.sh pre <run.yaml> [overrides]` | `./submit.sh` runs it; a FAIL stops the submission (`--no-checks` to override) | ~45 s, login node | the run would train or sample a network / tokenization / dataset that is not what you think |
| `./check.sh gpu <run.yaml>` | once per new or changed model dir, container or SkyRL version; prints the `sbatch` line | 1 node, ~20 min | vLLM and transformers disagree on log-probs for this model dir |
| `./check.sh run <run dir or slurm log> [--table N]` | after step 1 of a run, and at its end | seconds | step 1 already shows a broken stack; or errors / NaN |
| `./check.sh export <export dir> --base <start model dir>` | before evaluating an export | ~3 min | the bf16 copy lost the update, or the export's config / tokenizer differ from the start model |
| `./check.sh selftest` | after changing `checks/` or the container | ~4 min | a check no longer catches the bug it was written for |
| `./check.sh list` | | | every check with one line of description |

Results are `PASS`, `WARN`, `FAIL`, `SKIP` (with the reason). Only FAIL sets exit code 1; a WARN is
something to read once and decide on.

## The checks

**pre**
- `model.rope_parity`: transformers and vLLM build the same rotary tables from `config.json`.
- `model.config_keys`: rope keys only one of the two libraries reads (`attention_factor`, `attn_factor`, `apply_yarn_scaling`, ...).
- `tokenizer.golden`: tokenizer + chat template reproduce `<model dir>/golden_tokenization.jsonl`.
- `data.pregen`: pregenerated parquet: group sizes, mask and response lengths, sequence ends, vocabulary, scores, advantage balance.
- `settings.ref_temperature`: temperature other than 1.0 together with a reference worker.
- `settings.export_dtype`: is the planned update (lr x steps) large enough to survive a bf16 cast.
- `settings.gpu_parity_record`: result of the last `gpu` job for this model dir (WARN when there is none).

**gpu** (`checks/gpu_parity.sbatch`): per-token log-probs of the golden prompts in vLLM and in plain
transformers. Limits: mean abs diff 0.05, mean signed diff 0.01 nats/token.

**run**
- `run.step1_noise_floor`: pregenerated runs, step 1: policy vs reference log-prob diff and share of tokens outside the ratio band.
- `run.rollout_mismatch`: online runs, first 3 steps: trainer vs vLLM log-prob diff.
- `run.errors`, `run.drift`, `run.steps`.

**export**
- `export.delta`: cosine between the trained update and what a bf16 copy keeps; says which directory to evaluate.
- `export.rope`, `export.tokenizer`: the export's config and tokenizer files behave like the start model's.

Thresholds are constants at the top of each module, next to the measurement they come from.

## Preparing a model directory (once per model)

1. `python -m checks.make_golden --data <data with trusted token ids> --out <model dir>/golden_tokenization.jsonl`
   (inside the container). Trusted = token ids produced by the pipeline that trained or sampled the model.
2. `./check.sh pre <run.yaml>` and fix what FAILs. For TildeOpen that was, in the model dir:
   - `config.json`: delete `"attention_factor": 1.0` from `rope_scaling` and the top-level `"apply_yarn_scaling"` line;
   - `tokenizer_config.json`: `"add_prefix_space": true`.
3. `./check.sh gpu <run.yaml>`, submit the printed command, wait, `./check.sh pre` again.

Changing any small file of the model dir invalidates cached PASSes and the gpu record on purpose.

## Layout

    check.sh              entry point (container Python, no GPU)
    checks/runner.py      runs the checks of a stage, prints the verdicts
    checks/lib.py         Result, registry, context, cache (<root>/.check_cache)
    checks/model.py tokenizer.py data.py settings.py runlog.py export.py     the checks
    checks/gpu_parity.py gpu_parity.sbatch                                   the GPU job
    checks/make_golden.py                                                    builds the tokenizer ground truth
    checks/st.py          numpy reader for safetensors, bf16 arithmetic
    checks/tests/test_logic.py        synthetic inputs; runs anywhere with numpy + pyarrow + pytest
    checks/tests/test_known_bugs.py   the three original bugs against the real model dir (LUMI)

Adding a check: a function `fn(ctx) -> Result` with `@check("<stage>", "<area>.<name>")` in one of
the modules, a threshold constant with its source, and a test that makes it FAIL.

## Not covered

- FSDP sharding and sequence parallelism against plain transformers on GPU: only indirectly, through
  `run.step1_noise_floor` and `run.rollout_mismatch` at step 1 of a real run.
- Which checkpoint generated a pregenerated dataset: nothing in the parquet says so. `model.path`
  must point to it; `export.delta` warns when `--base` is clearly another checkpoint.
- Harbor environment plane, reward functions, evaluation harness.
