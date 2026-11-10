"""
Entrypoint: GRPO on pre-generated rollouts with SkyRL.

Everything algorithmic is a SkyRL built-in, selected in configs/pregen/grpo_bs5.yaml:
    trainer.algorithm.policy_loss_type=rollout_is      banded importance ratio, stop-gradient weight,
                                                       zero outside (1-eps_clip_low, 1+eps_clip_high)
    trainer.algorithm.use_kl_loss=true                 k3 KL to the reference worker (frozen start checkpoint)

What is custom here:
    PregeneratedGenerator / PregeneratedDataset   stored rollouts instead of sampling (generator.py)
    PregeneratedTrainer.train_critic_and_policy
                                      copies the reference worker's log-probs (base_action_log_probs,
                                      same FSDP/Ulysses stack as the policy) into rollout_logprobs, so the
                                      rollout_is ratio is pi_theta / pi_ref, exactly 1 at step 1
    NoInferenceClient + PregeneratedTrainer.init_weight_sync_state
                                      no vLLM engines are started and no weights are pushed: the
                                      inference node and the per-step broadcast are gone

Run (inside the container, under ray):
    python -m skyrl_extensions.pregenerated.entrypoint data.train_data="['/path/pregenerated.parquet']" <skyrl overrides>
"""
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from skyrl_extensions.pregenerated.generator import PregeneratedDataset, PregeneratedGenerator, PregeneratedStore

class NoInferenceClient:
    """
    Stands in for the vLLM engines, which are never started. Every method is an awaitable no-op:
    the trainer only calls sleep()/wake_up()/pause/resume on the client (trainer.py 381, 573), the
    generator never touches it, and the weight-sync group is skipped below. If a traceback shows
    the trainer reading a value from this object, add that one method explicitly.
    Private and dunder names are not answered: copy/pickle/hasattr probes must see a normal object.
    """

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _noop(*args, **kwargs):
            return None
        return _noop


class PregeneratedTrainer(RayPPOTrainer):
    def init_weight_sync_state(self):
        # trainer.py 855: would build the NCCL group between policy workers and engines. Instead,
        # disable the two pushes that follow it (initial sync at 289, per-step sync at 494), which
        # both call self.dispatch.save_weights_for_sampler(). Runs after self.dispatch exists.
        async def _no_sync(model_id=None):
            return None
        self.dispatch.save_weights_for_sampler = _no_sync

    def train_critic_and_policy(self, data):
        base = data.get("base_action_log_probs", None)
        if base is None:
            raise ValueError("rollout_is needs the reference worker's log-probs: set trainer.algorithm.use_kl_loss=true")
        data["rollout_logprobs"] = base
        return super().train_critic_and_policy(data)


class PregeneratedExp(BasePPOExp):
    def get_inference_client(self):
        # main_base.py 210: would start the vLLM servers (an extra node for a 30B model). Nothing
        # here generates, so no engine, no placement group, no weight sync.
        return NoInferenceClient()

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        alg = cfg.trainer.algorithm
        assert alg.policy_loss_type == "rollout_is" and alg.use_kl_loss, (
            "pregenerated training needs trainer.algorithm.policy_loss_type=rollout_is and use_kl_loss=true"
        )
        return PregeneratedGenerator(PregeneratedStore(cfg.data.train_data[0], max_seq_len=alg.max_seq_len))

    def get_train_dataset(self):
        return PregeneratedDataset(self.cfg.data.train_data[0], self.cfg.generator.n_samples_per_prompt)

    def get_eval_dataset(self):
        return None

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
        return PregeneratedTrainer(
            cfg=cfg, tracker=tracker, tokenizer=tokenizer, train_dataset=train_dataset,
            eval_dataset=eval_dataset, inference_engine_client=inference_engine_client,
            generator=generator, colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    PregeneratedExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must cover the longest prompt + response in the parquet")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()