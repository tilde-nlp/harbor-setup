"""
Pregenerated-rollout generator: hands SkyRL's trainer stored, pre-tokenized rollouts instead of sampling.

Parquet, one row per rollout:
    uid, rep, prompt (json conversation), prompt_ids, response_ids, loss_mask, score (0..SCORE_MAX),
    stop_reason, optional ref_logprobs (placeholder only; see PregeneratedTrainer in entrypoint.py).

Interfaces, checked against skyrl/train/generators/base.py and generators/utils.py (main, Oct 2026):
    prepare_generator_input() repeats each dataset item n_samples_per_prompt times, contiguously,
    and builds TrajectoryID(instance_id=item["uid"], repetition_id=0..n-1). The generator looks
    up (uid, repetition_id) in the parquet. GRPO groups by uid.
"""
import json
from collections import Counter

import pyarrow.parquet as pq
from loguru import logger

from skyrl.train.generators.base import GeneratorInput, GeneratorInterface, GeneratorOutput

SCORE_MAX = 10.0  # stored scores are 0..10; GRPO sees 0..1 (the scale the KL coefficient was chosen for)


class PregeneratedStore:
    """All rollouts of the parquet as one Arrow table, indexed by (uid, rep)."""

    def __init__(self, path: str):
        self.table = pq.read_table(path)
        keys = zip(self.table.column("uid").to_pylist(), self.table.column("rep").to_pylist())
        self.index = {k: i for i, k in enumerate(keys)}
        self.has_logprobs = "ref_logprobs" in self.table.column_names
        logger.info(f"PregeneratedStore: {len(self.index)} rollouts from {path}")

    def get(self, uid: str, rep: int) -> dict:
        return self.table.slice(self.index[(uid, rep)], 1).to_pylist()[0]


class PregeneratedDataset:
    """One item per prompt, in the shape BasePPOExp / prepare_generator_input expect (as the Harbor dataset)."""

    def __init__(self, path: str, n_samples_per_prompt: int):
        t = pq.read_table(path, columns=["uid", "rep", "prompt"])
        uids, reps, prompts = (t.column(c).to_pylist() for c in ("uid", "rep", "prompt"))
        bad = [u for u, c in Counter(uids).items() if c != n_samples_per_prompt]
        assert not bad, f"{len(bad)} prompts without exactly {n_samples_per_prompt} rollouts, e.g. {bad[:3]}"
        self.items = [
            {"prompt": json.loads(p), "env_class": None, "env_extras": {"uid": u}, "uid": u}
            for u, r, p in zip(uids, reps, prompts)
            if r == 0
        ]
        logger.info(f"PregeneratedDataset: {len(self.items)} prompts x {n_samples_per_prompt} rollouts")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        yield from self.items

    def collate_fn(self, items):
        return items


class PregeneratedGenerator(GeneratorInterface):
    def __init__(self, store: PregeneratedStore):
        self.store = store

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        tids = input_batch["trajectory_ids"]
        rows = [self.store.get(t.instance_id, t.repetition_id) for t in tids]
        out: GeneratorOutput = {
            "prompt_token_ids": [r["prompt_ids"] for r in rows],
            "response_ids": [r["response_ids"] for r in rows],
            "rewards": [float(r["score"]) / SCORE_MAX for r in rows],
            "loss_masks": [[int(m) for m in r["loss_mask"]] for r in rows],
            "stop_reasons": [r["stop_reason"] for r in rows],
            "rollout_metrics": {"generate/pregenerated_rows": len(rows)},
            # Placeholder: PregeneratedTrainer replaces rollout_logprobs with the reference worker's
            # log-probs before the loss runs. Only needed if validate_generator_output insists
            # on the field being present for rollout_is; None otherwise.
            "rollout_logprobs": [r["ref_logprobs"] for r in rows] if self.store.has_logprobs else None,
            "trajectory_ids": tids,
        }
        for k in (
            "trajectory_generation_times", "trajectory_time_splits", "rollout_expert_indices",
            "rollout_sample_support", "is_last_step", "env_metrics", "pixel_values", "image_grid_thw",
        ):
            out[k] = None
        return out