# SkyRL on LUMI

```
mkdir <dir> && cd <dir> && git clone <repo url> . && ./setup.sh [<from_root>]   # once: clone = working root
./submit.sh configs/<task>/<setup>.yaml [key.path=value ...]                      # submit
./submit.sh --dry-run configs/<task>/<setup>.yaml                                 # print resolved config + SkyRL args
```

| path | what goes there |
|---|---|
| `configs/` | one dir per task (`single_step`, `benchmark`, `harbor`, `pregen`), one self-contained YAML per setup with what it sets. `REFERENCE.yaml`: what every key means, with SkyRL's defaults |
| `launch/` | job machinery, rarely edited: `render.py` (config → run dir, checks), `job.sbatch` (training job), `harbor_env.sbatch` (Harbor workers), `lumi_env.sh` (every env var, with why) |
| `skyrl_extensions/` | Python added to SkyRL; each subpackage has an `entrypoint.py` selected by `entrypoint.module` in a config. `pregenerated/`: GRPO on stored rollouts. `judge/`: on-policy single-step RL with the reward from a judge model on its own node (`configs/single_step/gsm8k_judge.yaml`, config section `judge:`) |
| `setup/` | building the container: install steps, patches, versions, lock |
| `setup.sh` | makes this clone a working root: links container, model, `hf/` from an existing root, copies `data/` (not Harbor yet) |
| `tools/` | after a run: `collect_metrics.py [job id \| run dir \| .out file \| --all]` turns the Slurm log into `metrics.csv` (one row per step) and `metrics.runs.csv` (one row per run, with the config); default is the latest run |
| `notes/` | bugs and experiments worth remembering |

The clone is the working root (`paths.root`); everything `setup.sh` and the runs add is git-ignored. Every submission gets `previous_runs/<timestamp>-<run>/`: resolved config, SkyRL args, code snapshot, Slurm and NCCL logs. The layout before this one: `git checkout pre-restructure`.
