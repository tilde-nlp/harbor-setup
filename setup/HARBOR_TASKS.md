

### DOWNLOAD DATASET

```bash
ROOT=/scratch/project_465003062/environment/containers/fresh_skyrl_fsdp
mkdir -p $ROOT/tasks
singularity exec -B /pfs,/scratch $ROOT/multitorch_sandbox bash -c "
  . /opt/venv/bin/activate
  cd $ROOT/tasks
  harbor datasets download reasoning-gym/reasoning-gym-easy
"
ls $ROOT/tasks
find $ROOT/tasks -name instruction.md | wc -l      # expect 288
```


### CONVERT TO SINGULARITY (NOT NEEDED WITH harbor fork, IT CONVERTS automatically)

In this particular dataset every task has the same environment. Run this to make sure that that's the case:
```bash

ROOT=/scratch/project_465003062/environment/containers/fresh_skyrl_fsdp
T=$(find $ROOT/tasks -maxdepth 2 -name 'reasoning-gym-*' -type d | head -1 | xargs dirname)   # the dataset dir
echo "dataset dir: $T"; ls $T | wc -l

echo "=== distinct Dockerfiles"
md5sum $T/*/environment/Dockerfile | awk '{print $1}' | sort | uniq -c
```