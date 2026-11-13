"""
Entrypoint: on-policy single-step RL where the reward comes from a judge model (env "judge", env.py).

Everything about training is SkyRL's main_base: the generator samples from the policy's vLLM engines, GRPO
as configured. Like SkyRL's examples/train/llm_as_a_judge/main_llm_judge.py, this file only registers the env.
Before any model is loaded it waits for the judge servers and checks them, so a judge that does not come up
costs minutes, not a loaded policy.

Rows of the dataset choose their env themselves (env_class column): "judge" rows are judged, rows of another
env (e.g. the gsm8k validation set, env_class "gsm8k") keep their rule-based reward.

Run (inside the container, under ray; ./submit.sh does this):
    python -m skyrl_extensions.judge.entrypoint <skyrl overrides>
"""
import os
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # In the task, not in main(): the generator (and with it every env) lives in this process.
    from skyrl_gym.envs import register

    from skyrl_extensions.judge import env

    env.connect(wait_s=int(os.environ.get("JUDGE_STARTUP_TIMEOUT_S", "1800")))
    register(id="judge", entry_point="skyrl_extensions.judge.env:JudgeEnv")
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.generator.batched:
        raise ValueError("generator.batched=true runs the judge requests one at a time: set it to false")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
