
### INITIAL SETUP

Inputs, all in this repo's `setup/`: `versions.env` (base SIF, SkyRL commit), `overrides.txt`
(uv overrides), `patches/skyrl/*.patch` (applied to the SkyRL checkout), `patches/marin_harbor/`
(applied to the marin-harbor checkout), `container.lock` (pip freeze of the working container).

```bash
cd /scratch/project_465003062/environment/containers/
mkdir fresh_skyrl_fsdp # use any name. this will be a directory with everything installed
cd fresh_skyrl_fsdp
sbatch <repo>/setup/build_sandbox.sbatch     # builds ./multitorch_sandbox from the SIF in versions.env
```

### ENTERING THIS BIH

```bash
export SANDBOX=/scratch/project_465003062/environment/containers/fresh_skyrl_fsdp/multitorch_sandbox
mkdir -p "$SANDBOX"/{users,project,projappl,flash,scratch,appl,pfs,var/spool/slurmd}
# enter
srun --account=project_465003062 --partition=small -N1 -n1 -c16 --mem=64G -t 4:00:00 --pty \
  singularity shell --writable \
  -B /pfs,/scratch,/projappl,/project,/flash,/appl,/users,/var/spool/slurmd \
  /scratch/project_465003062/environment/containers/fresh_skyrl_fsdp/multitorch_sandbox


# env
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
IS=<repo>/setup     # this repo, cloned under /scratch
SKYRL_COMMIT=59d4daedee24c7b1a79d857bbf322a6d195c3792


# 0. baseline of the untouched container
pip freeze > /opt/pre.txt
python - <<'EOF'
from importlib.metadata import distributions
lines = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata['Name']})
open('/opt/constraints.txt','w').write("\n".join(lines)+"\n"); print(len(lines))
EOF
pip show peft debugpy transformers vllm torch | grep -E '^(Name|Version)'

# 1. SkyRL source at the pinned commit, AMD pyproject, patches
mkdir -p /opt/src && cd /opt/src
git clone https://github.com/NovaSky-AI/SkyRL.git
cd /opt/src/SkyRL
git checkout "$SKYRL_COMMIT"
cp docker/pyproject.amd.toml pyproject.toml
for p in "$IS"/patches/skyrl/*.patch; do git apply --check "$p" && git apply "$p" && echo "applied $(basename $p)"; done
git status --short

# 2. resolver
pip install uv
U="uv pip install --python /opt/venv/bin/python -c /opt/constraints.txt"

# 3. dry runs; all four must finish without errors
$U --dry-run "ray[default]==2.57.0" orjson torchdata
$U --dry-run https://github.com/SumanthRH/router/releases/download/0.1.14.post1/vllm_router-0.1.14-cp38-abi3-manylinux_2_35_x86_64.whl
$U --dry-run -e /opt/src/SkyRL/skyrl-gym
$U --dry-run --overrides "$IS/overrides.txt" --no-build-isolation -e "/opt/src/SkyRL[fsdp]"

# 4. real installs
$U "ray[default]==2.57.0" orjson torchdata
$U https://github.com/SumanthRH/router/releases/download/0.1.14.post1/vllm_router-0.1.14-cp38-abi3-manylinux_2_35_x86_64.whl
$U -e /opt/src/SkyRL/skyrl-gym
$U --overrides "$IS/overrides.txt" --no-build-isolation -e "/opt/src/SkyRL[fsdp]"

# 5. verify
python - <<'EOF'
import importlib.metadata as m
import torch, vllm, ray, skyrl, vllm_router, skyrl_gym
from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend
assert int(m.version("setuptools").split(".")[0]) < 80, m.version("setuptools")
print("ok torch", torch.__version__, "vllm", vllm.__version__, "ray", ray.__version__, "transformers", m.version("transformers"))
EOF
pip freeze > /opt/post.txt
diff /opt/pre.txt /opt/post.txt
pip check || true

# 6. record
pip freeze > /scratch/project_465003062/environment/containers/fresh_skyrl_fsdp/new.lock
exit


# IN THE LOGIN NODE
cd /scratch/project_465003062/environment/containers/fresh_skyrl_fsdp
old=<repo>/setup/container.lock    # replace it with new.lock once the new container works
echo "=== in old only (<) / in new only (>) / version differs (both)"
diff <(grep -v ' @ ' $old | grep -v '^-e ' | sort) <(grep -v ' @ ' new.lock | grep -v '^-e ' | sort)
echo "=== repo/path installs"
grep -E ' @ |^-e ' $old new.lock

```



### CORRECTIONS TO MY MISTAKES


```bash

uv pip install --python /opt/venv/bin/python "transformers==5.8.0" "peft==0.18.1"
uv pip install --python /opt/venv/bin/python "vllm-router==0.1.15"

```


### TRAINING SETUP (gsm8k first)


```bash
cd $NEW
HF_HOME=$NEW/hf singularity exec -B /pfs,/scratch multitorch_sandbox \
  bash -c '. /opt/venv/bin/activate; hf download Qwen/Qwen3-14B'
ls hf/hub/models--Qwen--Qwen3-14B/snapshots/*/ | head

NEW=/scratch/project_465003062/environment/containers/fresh_skyrl_fsdp
cd $NEW
HF_HOME=$NEW/hf singularity exec -B /pfs,/scratch multitorch_sandbox \
  bash -c '. /opt/venv/bin/activate; python /opt/src/SkyRL/examples/train/gsm8k/gsm8k_dataset.py --output_dir '"$NEW"'/data'
ls -la $NEW/data
```


### HARBOR INSTALL

```bash


```