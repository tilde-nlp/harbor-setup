# Guide for using SkyRL with Harbor on LUMI


## INSTALLATION

### Overview
We're gonna take the laifs/multitorch container (/appl/local/laifs/containers) and modify it via sandbox. We will need to install ray, uv and tinker (for smoke test) packages and skyrl repo. After that, install harbor datasets, and patch them to fit with singularity. <br>

All of the steps are done in the /scratch/project_465003062/environment/containers/skyrl-harbor directory.

### 0. Prepare
```bash
# copy all of the scripts to the $ROOT/scripts directory

# make all of them executable
chmod +x scripts/*
```

### 1. Unbox laifs container
I am using lumi-multitorch-u24r70f21m50t210-20260807_115122 in this guide, it's the newest at the time.

THIS WAS A MISTAKE, USE lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif INSTEAD. IT'S ALSO THE NEWEST, BUT HAS transformer_engine

```bash
# symlink 
ln -s /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif .
mv lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif lumi-multitorch-plus.sif

# build sandbox via srun (faster and more polite than doing so in login)
./scripts/1_build_sandbox.sh
```

### 2. Install SkyRL repo

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git SkyRL
rm -rf SkyRL/.git
mkdir -p multitorch_sandbox/opt/SkyRL
cp -a SkyRL multitorch_sandbox/opt/SkyRL
```

### 3. Enter the sandbox

```bash
# create directories needed for sif files
cd multitorch_sandbox
mkdir users project projappl flash scratch appl pfs
cd ..

# enter
singularity shell --writable multitorch_sandbox/

# activate env
. /opt/venv/bin/activate
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
```

### 4. Dry runs (to check if installing is gonna break anything)

```bash
cd /opt
python - <<'EOF'
from importlib.metadata import distributions
lines = sorted({f"{d.metadata['Name']}=={d.version}"
                for d in distributions() if d.metadata['Name']})
open('/opt/constraints.txt','w').write("\n".join(lines)+"\n")
print(len(lines))
EOF
cp /opt/constraints.txt /opt/pre.txt

# using amd.pyproject instead of the basic
cd /opt/SkyRL/SkyRL
cp docker/pyproject.amd.toml pyproject.toml


# ADD MEGATRON EXTRA TO THE AMD PYPROJECT COPY, NEXT TO THE FSDP
# THIS SHOULD BE ADDED TO DHE PYPROJECT.TOML FILE IN THE SKYRL ROOT
megatron = [
    "skyrl[skyrl-train]",
]

# uv shouldn't have any problems installing 
pip install uv

# override, because laifs got transformers==5.10.1, but skyrl depends on transformers<=5.8.0 (and some other bs)
# we're taking the laifs version because it's compiled against lumi hardware
cat > /opt/overrides.txt <<'EOF'
transformers>=5.6.1
peft>=0.18.1
debugpy>=1.8.0
EOF

# dry runs
uv pip install --python /opt/venv/bin/python --dry-run \
  -c /opt/constraints.txt "ray[default]==2.56.0" orjson torchdata

uv pip install --python /opt/venv/bin/python --dry-run \
  -c /opt/constraints.txt -e /opt/SkyRL/SkyRL/skyrl-gym

uv pip install --python /opt/venv/bin/python --dry-run \
  -c /opt/constraints.txt --overrides /opt/overrides.txt --no-build-isolation \
  -e "/opt/SkyRL/SkyRL[megatron,tinker]"


### INSTALLING MBRIDGE AND MCORE NOW
# exit the apptainer for now
exit 

# host: stage sources into sandbox
git clone https://github.com/NVIDIA/Megatron-LM mcore
git -C mcore checkout 71e418ea7d7b3a6c9a53238c543c3e0b43e11026
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge mbridge
git -C mbridge checkout 91a15142a4b4442a8d46ab539d1b923bd08570d0
rm -rf mcore/.git mbridge/.git
mkdir -p multitorch_sandbox/opt/src
cp -a mcore mbridge multitorch_sandbox/opt/src/

# enter the container now, step 3

# dry runs, last time i promise:
uv pip install --python /opt/venv/bin/python --dry-run \
  -c /opt/constraints.txt --overrides /opt/overrides.txt --no-build-isolation \
  -e "/opt/SkyRL/SkyRL[megatron,tinker]"

uv pip install --python /opt/venv/bin/python --dry-run --no-deps --no-build-isolation \
  /opt/src/mcore
uv pip install --python /opt/venv/bin/python --dry-run --no-deps --no-build-isolation \
  /opt/src/mbridge

uv pip install --python /opt/venv/bin/python --dry-run -c /opt/constraints.txt \
  einops mistral-common qwen-vl-utils

# ALL OF THE DRY RUNS SHOULD RUN WITHOUT ERRORS
### REAL INSTALLS!!!

uv pip install --python /opt/venv/bin/python \
  -c /opt/constraints.txt "ray[default]==2.56.0" orjson torchdata

uv pip install --python /opt/venv/bin/python \
  -c /opt/constraints.txt -e /opt/SkyRL/SkyRL/skyrl-gym

uv pip install --python /opt/venv/bin/python \
  -c /opt/constraints.txt --overrides /opt/overrides.txt --no-build-isolation \
  -e "/opt/SkyRL/SkyRL[megatron,tinker]"

# swap megatron-core
# explanation: we need megatron, but laifs already packaged it. with old version though.
# plus we need megatron-bridge

pip uninstall -y megatron-core
uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation /opt/src/mcore
python -c "import megatron.core, megatron.core._rank_utils; print(megatron.core.__version__)" # check

# install megatron-bridge
uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation /opt/src/mbridge
uv pip install --python /opt/venv/bin/python -c /opt/constraints.txt \
  einops mistral-common qwen-vl-utils

# also need modelopt
uv pip install --python /opt/venv/bin/python --no-deps \
  -c /opt/constraints.txt "nvidia-modelopt==0.44.0"
uv pip install --python /opt/venv/bin/python \
  -c /opt/constraints.txt "pulp<4.0" nvidia-ml-py
python -c "import megatron.bridge; print(megatron.bridge.__version__)" # check


### WE ALSO NEED TRANSFORMERENGINE WTFFFF
# host
git clone --recursive https://github.com/ROCm/TransformerEngine te
rm -rf te/.git
cp -a te multitorch_sandbox/opt/src/te

salloc -A project_465003062 -p small -N1 -n1 -c32 --mem=96G -t 4:00:00
srun --pty bash

# inside sandbox
cd /opt/src/te
export NVTE_FRAMEWORK=pytorch
export NVTE_ROCM_ARCH=gfx90a PYTORCH_ROCM_ARCH=gfx90a
pip install --no-deps --no-build-isolation -v .
python -c "import transformer_engine.pytorch as te; import transformer_engine; print(transformer_engine.__version__)"



# veryfying that we installed everything

python - <<'EOF'
import importlib.metadata as m
import torch, vllm, ray, tinker, skyrl, skyrl.tinker.api
import megatron.core, megatron.core._rank_utils, megatron.bridge
from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend, MegatronBackendOverrides
assert int(m.version("setuptools").split(".")[0]) < 80, m.version("setuptools")
print("ok", torch.__version__, vllm.__version__, megatron.core.__version__)
EOF
python - <<'EOF'
from importlib.metadata import distributions
lines = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})
open('/opt/post.txt','w').write("\n".join(lines)+"\n")
EOF
diff /opt/pre.txt /opt/post.txt   # additions + overrides + mcore swap only; torch/vllm/TE must not move
```


### 5. Smoke test (can skip)

```bash
# inside the container
