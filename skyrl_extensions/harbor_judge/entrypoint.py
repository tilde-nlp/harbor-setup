"""
Entrypoint: multi-step agentic RL on Harbor tasks with the reward from a judge model. SkyRL's Harbor entrypoint
(examples/train_integrations/harbor/entrypoints/main_harbor.py: config, dataset, trials, step-wise GRPO) with one
change: the generator is JudgedHarborGenerator (generator.py), which replaces each trial's verifier reward by the
judge's verdict on the agent's session. The judge: the config's `judge:` section, the same servers and client as
the single-step judge (skyrl_extensions/judge/), with a judge for Harbor trials (judges.py, e.g. harbor_task).

Before any model is loaded it waits for the judge servers and checks them, so a judge that does not come up costs
minutes, not a loaded policy.

Run (inside the container, under ray, cwd /opt/SkyRL for the examples package; ./submit.sh does this):
    python -m skyrl_extensions.harbor_judge.entrypoint <skyrl overrides>
"""
import os
import sys

import ray
import yaml

from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


class JudgedHarborExp(HarborExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from skyrl_extensions.harbor_judge.generator import JudgedHarborGenerator

        return JudgedHarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # In the task, not in main(): the generator, which asks the judge, lives in this process.
    from skyrl_extensions.judge import env
    from skyrl_extensions.judge.judges import JUDGES

    client = env.connect(wait_s=int(os.environ.get("JUDGE_STARTUP_TIMEOUT_S", "1800")))
    if not getattr(client.judge, "harbor", False):
        raise ValueError("judge.name={}: grades single completions, not Harbor trials. Judges for Harbor trials: {}".format(
            os.environ.get("JUDGE_NAME"), ", ".join(n for n, j in sorted(JUDGES.items()) if getattr(j, "harbor", False))))
    JudgedHarborExp(cfg).run()


def main() -> None:
    # main_harbor.main(), with this file's skyrl_entrypoint
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be explicitly set for Harbor training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
