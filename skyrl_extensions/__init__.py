"""
Extensions to SkyRL: code that runs inside the container, inside SkyRL's process, and adds what
stock SkyRL does not have. Nothing in the SkyRL checkout is modified.

How an extension is plugged in ("registered"). SkyRL has no plugin discovery, so:
  1. Each extension is a subpackage with an `entrypoint.py`: a copy of SkyRL's main_base entrypoint
     whose BasePPOExp subclass overrides the hooks it needs (get_generator, get_train_dataset,
     get_trainer, get_inference_client, ...). Things SkyRL does keep registries for (policy losses,
     advantage estimators) are registered from that entrypoint, before training starts.
  2. A config selects it:  entrypoint.module: skyrl_extensions.<name>.entrypoint
  3. The launcher puts the run's snapshot of this repo on PYTHONPATH (launch/lumi_env.sh), so
     `python -m skyrl_extensions.<name>.entrypoint` resolves in the driver and in every Ray actor.

Extensions:
  pregenerated   GRPO on stored, pre-tokenized rollouts instead of sampling (configs/pregen/)
"""
