"""
Entrypoint: GRPO on pre-generated rollouts with SkyRL.

Mirrors examples/train_integrations/harbor/entrypoints/main_harbor.py (main, Oct 2026):
BasePPOExp with get_generator / get_train_dataset / get_eval_dataset / get_trainer overridden.

Reference log-probs come from SkyRL's own reference worker (trainer.algorithm.use_kl_loss=true,
REF_NODES>=1 or colocate_policy_ref): every step the ref worker runs the frozen starting
checkpoint over the batch in the same FSDP/Ulysses stack as the policy, and the trainer
subclass below copies those values (`base_action_log_probs`) into `rollout_logprobs` before
training. replay/loss.py then uses them for the banded importance ratio against the behaviour
policy; the KL to the same reference is SkyRL's built-in use_kl_loss term.

Nothing is precomputed and no ref_logprobs column is needed. If the parquet has one anyway, it
is overwritten by the ref worker's values (a reference from another stack, e.g. vLLM, is off
by ~0.2 nats per token on this model at SP=4).

Run (inside the container, under ray):
    python -m replay.main_replay data.train_data="['/path/replay.parquet']" <other skyrl overrides>
"""
import sys

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

import replay.loss  # noqa: F401  registers the "replay_offline" policy loss before Ray starts
from replay.generator import ReplayDataset, ReplayGenerator, ReplayStore
from replay.loss import LOSS_NAME


class ReplayTrainer(RayPPOTrainer):
    """
    replay_offline + use_kl_loss: the reference worker's log-probs become the behaviour-policy
    reference of the importance ratio. Everything else is the stock SkyRL loop.
    """

    def _skip_policy_forward(self, training_input) -> bool:
        # replay_offline never reads old_log_probs: the ratio is against the ref worker's log-probs.
        # Skipping saves one no-grad policy forward per step. The ref worker's forward is separate.
        alg = self.cfg.trainer.algorithm
        if alg.policy_loss_type == LOSS_NAME and not alg.use_kl_in_reward:
            return True
        return super()._skip_policy_forward(training_input)

    def train_critic_and_policy(self, data):
        alg = self.cfg.trainer.algorithm
        if alg.policy_loss_type == LOSS_NAME:
            base = data.get("base_action_log_probs", None)
            if base is None:
                raise ValueError(
                    f"{LOSS_NAME} needs the reference worker's log-probs (base_action_log_probs); "
                    "set trainer.algorithm.use_kl_loss=true with REF_NODES>=1 or colocate_policy_ref=true"
                )
            if self.global_step == 1 and data.get("rollout_logprobs", None) is not None:
                logger.info(f"{LOSS_NAME}: replacing the parquet's rollout_logprobs with the ref worker's log-probs")
            data["rollout_logprobs"] = base
        return super().train_critic_and_policy(data)


class ReplayExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        # The inference client is not used: nothing is generated. SkyRL still starts the
        # engines and syncs weights to them each step; keep them small (see sbatch).
        store = ReplayStore(cfg.data.train_data[0])
        alg = cfg.trainer.algorithm
        if alg.policy_loss_type == LOSS_NAME and not alg.use_kl_loss:
            raise ValueError(f"policy_loss_type={LOSS_NAME} needs trainer.algorithm.use_kl_loss=true (the reference worker "
                             "supplies the behaviour-policy log-probs every step)")
        if store.has_ref_logprobs:
            logger.warning("the parquet has a ref_logprobs column; it is ignored, the ref worker's values are used instead")
        return ReplayGenerator(store)

    def get_train_dataset(self):
        ds = ReplayDataset(self.cfg.data.train_data, self.cfg.generator.n_samples_per_prompt)
        assert len(ds) >= self.cfg.trainer.train_batch_size, (
            f"dataset has {len(ds)} prompts, train_batch_size is {self.cfg.trainer.train_batch_size}"
        )
        return ds

    def get_eval_dataset(self):
        return None

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
        return ReplayTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # The training loop runs in a Ray task, not on the driver process.
    ReplayExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set (longest prompt + response in the parquet)")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()