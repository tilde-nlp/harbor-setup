# SkyRL on LUMI

```
./submit.sh configs/<task>/<run>.yaml [key.path=value ...]   # submit
./submit.sh --dry-run configs/<task>/<run>.yaml              # print resolved config + SkyRL args
```

| path | what goes there |
|---|---|
| `configs/` | one dir per task (`single_step`, `benchmark`, `harbor`, `pregen`), one self-contained YAML per setup with what it sets. `REFERENCE.yaml`: what every key means, with SkyRL's defaults |
| `launch/` | job machinery, rarely edited: `render.py` (config → run dir, checks), `job.sbatch` (training job), `harbor_env.sbatch` (Harbor workers), `lumi_env.sh` (every env var, with why) |
| `skyrl_extensions/` | Python added to SkyRL; each subpackage has an `entrypoint.py` selected by `entrypoint.module` in a config |
| `setup/` | building the container: install steps, patches, versions, lock |
| `setup.sh` | new working root (`paths.root`) from an existing one: links container, model, `hf/`, copies `data/` (not Harbor yet) |
| `notes/` | bugs and experiments worth remembering |

Every submission gets `<paths.runs>/<timestamp>-<run>/`: resolved config, SkyRL args, code snapshot, Slurm and NCCL logs. The layout before this one: `git checkout pre-restructure`.
