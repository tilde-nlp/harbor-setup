"""
Replay generator for SkyRL: returns stored, pre-tokenized rollouts instead of generating.

Interfaces (checked against skyrl/train/generators/base.py on main, Oct 2026):
    GeneratorInput  = {prompts, env_classes, env_extras, sampling_params, trajectory_ids, batch_metadata}
    TrajectoryID    = (instance_id: str, repetition_id: int)
    GeneratorOutput = {prompt_token_ids, response_ids, rewards, loss_masks, stop_reasons,
                       rollout_metrics, rollout_logprobs, trajectory_ids, ...optional fields}

`prepare_generator_input()` in skyrl/train/generators/utils.py repeats each dataset item
n_samples_per_prompt times, contiguously, and builds TrajectoryID(instance_id=item["uid"],
repetition_id=0..n-1). So the dataset item's "uid" must be the parquet "uid", and the
generator looks up (uid, repetition_id).
"""
import json
import os
from collections import Counter
from typing import Any, Dict, List

import pyarrow.parquet as pq
from loguru import logger

from skyrl.train.generators.base import GeneratorInput, GeneratorInterface, GeneratorOutput


class ReplayStore:
    """All rollouts of the parquet, kept as an Arrow table, indexed by (uid, rep)."""

    def __init__(self, path: str):
        self.table = pq.read_table(path)
        uids = self.table.column("uid").to_pylist()
        reps = self.table.column("rep").to_pylist()
        self.index = {(u, r): i for i, (u, r) in enumerate(zip(uids, reps))}
        self.reps_per_uid = Counter(uids)
        self.has_ref_logprobs = "ref_logprobs" in self.table.column_names
        logger.info(
            f"ReplayStore: {len(self.index)} rollouts, {len(self.reps_per_uid)} prompts, "
            f"ref_logprobs={'yes' if self.has_ref_logprobs else 'no'} ({path})"
        )

    def get(self, uid: str, rep: int) -> Dict[str, Any]:
        return self.table.slice(self.index[(uid, rep)], 1).to_pylist()[0]


class ReplayDataset:
    """
    One item per prompt. Same item shape as examples/train_integrations/harbor/dataset.py:
    {"prompt", "env_class", "env_extras", "uid"} plus collate_fn returning the list.
    """

    def __init__(self, data_files: List[str], n_samples_per_prompt: int):
        assert len(data_files) == 1, "replay dataset takes exactly one parquet"
        table = pq.read_table(data_files[0], columns=["uid", "rep", "prompt"])
        uids = table.column("uid").to_pylist()
        reps = table.column("rep").to_pylist()
        prompts = table.column("prompt").to_pylist()
        counts = Counter(uids)
        bad = [u for u, c in counts.items() if c != n_samples_per_prompt]
        assert not bad, f"{len(bad)} prompts do not have exactly {n_samples_per_prompt} rollouts, e.g. {bad[:3]}"
        seen = set()
        self.items = []
        for uid, rep, prompt in zip(uids, reps, prompts):
            if rep != 0 or uid in seen:
                continue
            seen.add(uid)
            self.items.append(
                {
                    "prompt": json.loads(prompt),
                    "env_class": None,
                    "env_extras": {"uid": uid},
                    "uid": uid,
                }
            )
        logger.info(f"ReplayDataset: {len(self.items)} prompts x {n_samples_per_prompt} rollouts")

    def __getitem__(self, index: int) -> dict:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        yield from self.items

    def collate_fn(self, item_list):
        return item_list


class ReplayGenerator(GeneratorInterface):
    def __init__(self, store: ReplayStore):
        self.store = store
        # REPLAY_REWARD_SCALE: multiply the stored score before it enters GRPO (scores are 0..10;
        # 0.1 maps them to 0..1 so the advantages are on the scale the KL coefficient was chosen for)
        self.reward_scale = float(os.environ.get("REPLAY_REWARD_SCALE", "1.0"))
        logger.info(f"ReplayGenerator: reward scale {self.reward_scale}")

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        tids = input_batch.get("trajectory_ids")
        extras = input_batch.get("env_extras")
        n = len(prompts)

        prompt_token_ids, response_ids, loss_masks, rewards, stop_reasons = [], [], [], [], []
        rollout_logprobs = [] if self.store.has_ref_logprobs else None

        fallback_rep = Counter()  # only used if the trainer passes no trajectory_ids
        for i in range(n):
            if tids is not None:
                uid, rep = tids[i].instance_id, tids[i].repetition_id
            else:
                uid = extras[i]["uid"]
                rep = fallback_rep[uid]
                fallback_rep[uid] += 1
            row = self.store.get(uid, rep)
            prompt_token_ids.append(row["prompt_ids"])
            response_ids.append(row["response_ids"])
            loss_masks.append([int(m) for m in row["loss_mask"]])
            rewards.append(float(row["score"]) * self.reward_scale)
            stop_reasons.append(row["stop_reason"])
            if rollout_logprobs is not None:
                rollout_logprobs.append(row["ref_logprobs"])

        out: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": {"generate/replay_rows": n},
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": tids,
            # Optional fields in GeneratorOutput, not used here.
            "trajectory_generation_times": None,
            "trajectory_time_splits": None,
            "rollout_expert_indices": None,
            "rollout_sample_support": None,
            "is_last_step": None,
            "env_metrics": None,
            "pixel_values": None,
            "image_grid_thw": None,
        }
        return out