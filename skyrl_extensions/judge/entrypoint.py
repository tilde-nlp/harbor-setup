"""
Entrypoint: on-policy single-step RL where the reward comes from a judge model (env "judge", env.py).

Everything about training is SkyRL's main_base: the generator samples from the policy's vLLM engines, GRPO
as configured. Like SkyRL's examples/train/llm_as_a_judge/main_llm_judge.py, this file only registers the env.
Before any model is loaded it waits for the judge servers and checks them, so a judge that does not come up
costs minutes, not a loaded policy.

Rows of the dataset choose their env themselves (env_class column): "judge" rows are judged, rows of another
env (e.g. the gsm8k validation set, env_class "gsm8k") keep their rule-based reward.

Optional DAPO soft overlong punishment (trainer.algorithm.overlong_buffer_len > 0; off by default), the same
rule as SkyRL's examples/train/algorithms/dapo/main_dapo.py: with L = generator.sampling_params.
max_generate_length and B = overlong_buffer_len, a response of n > L - B tokens loses
(n - (L - B)) / B * overlong_buffer_penalty_factor of its reward, up to the whole factor at n = L. Not
main_dapo itself: it would drop the judge env, the judge server check and SpillPolicyWorker.

Run (inside the container, under ray; ./submit.sh does this):
    python -m skyrl_extensions.judge.entrypoint <skyrl overrides>
"""
import os
import sys
from dataclasses import dataclass

import ray
import torch

from skyrl.train.config import AlgorithmConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


@dataclass
class JudgeAlgorithmConfig(AlgorithmConfig):
    overlong_buffer_len: int = 0              # 0 = no overlong punishment (DAPO example: 512)
    overlong_buffer_penalty_factor: float = 1.0


JudgeConfig = make_config(algorithm_cls=JudgeAlgorithmConfig)


def overlong_penalty(response_length, max_length, buffer_len, factor):
    """DAPO soft overlong punishment: 0 up to max_length - buffer_len, then linear up to factor."""
    exceed = response_length - (max_length - buffer_len)
    if exceed <= 0:
        return 0.0
    return min(exceed, buffer_len) / buffer_len * factor


class JudgeTrainer(RayPPOTrainer):
    @torch.no_grad()
    def postprocess_generator_output(self, generator_output, uids, *args, **kwargs):
        algo = self.cfg.trainer.algorithm
        if algo.overlong_buffer_len > 0:
            rewards = generator_output["rewards"]
            assert not isinstance(rewards[0], list), "the overlong punishment needs one reward per rollout"
            max_length = self.cfg.generator.sampling_params.max_generate_length   # single-step: the response cap
            generator_output["rewards"] = [
                r - overlong_penalty(len(ids), max_length, algo.overlong_buffer_len, algo.overlong_buffer_penalty_factor)
                for r, ids in zip(rewards, generator_output["response_ids"])
            ]
        return super().postprocess_generator_output(generator_output, uids, *args, **kwargs)

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        # With a reference on the policy GPUs (use_kl_loss, layout.ref_nodes 0) the workers offload to host RAM,
        # and SkyRL then binds each worker's memory strictly to one NUMA node: rank 0 cannot gather the model
        # for the HF export (host OOM, notes/BUGS.md). SpillPolicyWorker lets the memory spill over; its export
        # also copies the start model's tokenizer files. Same worker as the pregenerated runs.
        if self.cfg.trainer.strategy == "fsdp":
            from skyrl_extensions.pregenerated.entrypoint import SpillPolicyWorker
            PolicyWorker = SpillPolicyWorker
        return super().build_models(PolicyWorker, CriticWorker, RefWorker)


class JudgeExp(BasePPOExp):
    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
        return JudgeTrainer(
            cfg=cfg, tracker=tracker, tokenizer=tokenizer, train_dataset=train_dataset,
            eval_dataset=eval_dataset, inference_engine_client=inference_engine_client,
            generator=generator, colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # In the task, not in main(): the generator (and with it every env) lives in this process.
    from skyrl_gym.envs import register

    from skyrl_extensions.judge import env

    env.connect(wait_s=int(os.environ.get("JUDGE_STARTUP_TIMEOUT_S", "1800")))
    register(id="judge", entry_point="skyrl_extensions.judge.env:JudgeEnv")
    JudgeExp(cfg).run()


def main() -> None:
    cfg = JudgeConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.generator.batched:
        raise ValueError("generator.batched=true runs the judge requests one at a time: set it to false")
    if cfg.trainer.algorithm.overlong_buffer_len > cfg.generator.sampling_params.max_generate_length:
        raise ValueError("trainer.algorithm.overlong_buffer_len is longer than max_generate_length")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
