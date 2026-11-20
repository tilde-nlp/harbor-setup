# Harbor install (marin-community/harbor)

marin-community/harbor is **not** installed as a package. It is used in two places:

| where | what | used by |
|---|---|---|
| `<root>/marin-harbor` (git checkout on LUMI) | `src/harbor/environments/apptainer/server.py`, `worker.py` | the bridge (`launch/harbor_bridge.py`, `launch/harbor_env.sbatch`), run with `module load cray-python`; configs point at it via `harbor_env.apptainer_dir: ${paths.root}/marin-harbor/src/harbor/environments/apptainer` |
| container `/opt/venv` | upstream `harbor 0.13.1` + marin-harbor's `apptainer` environment copied in | the SkyRL trials (`harbor_trial_config.environment.type=apptainer`, an HTTP client of the bridge) |

Versions:

* marin-harbor: `https://github.com/marin-community/harbor.git` @ `abdd67137bcd6e5c44d682d0052fbd1190e63556`
  ("Publish Harbor benchmark metadata (#153)"), plus `setup/patches/marin_harbor/0001-even-worker-distribution.patch`
  (changes only `server.py` and `worker.py`).
* upstream harbor in the container: `harbor @ git+https://github.com/laude-institute/harbor@3de07a0e01f3368921766437fc7afece3ddec23d`
  (0.13.1, no extras; see `container.lock`).

## How the original was done (fsdp-skyrl-harbor, 2026-09)

Reconstructed from timestamps, git reflog and the wheel's RECORD hashes:

1. 2026-09-16: cloned marin-community/harbor into `fsdp-skyrl-harbor/marin-harbor` (HEAD `abdd6713`).
2. 2026-09-16: installed upstream harbor @ `3de07a0` into the sandbox's `/opt/venv` with uv.
3. Copied `marin-harbor/src/harbor/environments/apptainer/` into `site-packages/harbor/environments/` and registered it
   by hand in two upstream files (the only files that differ from the wheel):
   * `harbor/models/environment_type.py`: `APPTAINER = "apptainer"` (after `DAYTONA`)
   * `harbor/environments/factory.py`: an `EnvironmentType.APPTAINER: _EnvEntry("harbor.environments.apptainer", "ApptainerEnvironment", None)` entry
4. 2026-09-29: applied the even-worker-distribution patch to the checkout (= `patches/marin_harbor/0001-...patch`)
   and copied the patched `server.py` / `worker.py` into the sandbox again.
5. 2026-10-02: sandbox built into `multitorch_sandbox.sif`. The SIF that `setup.sh` hard-links into a new root is this
   one, so **the container side is already done** for any root made with `setup.sh`.

## New root: the LUMI checkout (needed, `setup.sh` does not do it)

```bash
cd <root>      # e.g. /scratch/project_465003062/environment/containers/skyrl-training

# A: reproducible
git clone https://github.com/marin-community/harbor.git marin-harbor
git -C marin-harbor checkout abdd67137bcd6e5c44d682d0052fbd1190e63556
git -C marin-harbor apply "$PWD/setup/patches/marin_harbor/0001-even-worker-distribution.patch"

# B: quickest, copy the working checkout (same state as A)
cp -a /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/marin-harbor .

# check
ls marin-harbor/src/harbor/environments/apptainer/server.py
git -C marin-harbor status --short    # only server.py and worker.py modified
```

`marin-harbor` is git-ignored (the `.gitignore` is a whitelist).

## Verify the container

```bash
singularity exec <root>/multitorch_sandbox.sif bash -c '. /opt/venv/bin/activate
python -c "import importlib.metadata as m, harbor.environments.apptainer as a; print(m.version(\"harbor\"), a.__file__)"
md5sum /opt/venv/lib/python3.12/site-packages/harbor/environments/apptainer/{server,worker}.py'
md5sum <root>/marin-harbor/src/harbor/environments/apptainer/{server,worker}.py   # must match the two above
```

## Fresh container build (after step 4 of INSTALL.md)

Not yet run as written. Inside the writable sandbox, with `/opt/constraints.txt` from step 0 and the patched
checkout from above:

```bash
uv pip install --python /opt/venv/bin/python -c /opt/constraints.txt \
  "harbor @ git+https://github.com/laude-institute/harbor@3de07a0e01f3368921766437fc7afece3ddec23d"

H=/opt/venv/lib/python3.12/site-packages/harbor
cp -r <root>/marin-harbor/src/harbor/environments/apptainer "$H/environments/"

sed -i 's/^    DAYTONA = "daytona"$/&\n    APPTAINER = "apptainer"/' "$H/models/environment_type.py"
python - "$H/environments/factory.py" <<'EOF'
import sys; p = sys.argv[1]; s = open(p).read()
anchor = '    EnvironmentType.DAYTONA: _EnvEntry('
assert 'EnvironmentType.APPTAINER' not in s and anchor in s
s = s.replace(anchor, '    EnvironmentType.APPTAINER: _EnvEntry(\n        "harbor.environments.apptainer",\n'
              '        "ApptainerEnvironment",\n        None,\n    ),\n' + anchor, 1)
open(p, 'w').write(s)
EOF

python -c "from harbor.environments.factory import *; import harbor.environments.apptainer; print('ok')"
```

## Keep both copies in sync

If `server.py` / `worker.py` change again (new patch in `patches/marin_harbor/`), update **both** the LUMI checkout
and the container's `site-packages/harbor/environments/apptainer/`: the bridge and workers run from the checkout, the
trials' client comes from the container.