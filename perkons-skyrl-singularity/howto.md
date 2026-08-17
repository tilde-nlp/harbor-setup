### 0. Make sure that Singularity/Apptainer are installed & you have fakeroot perms

```bash
# Singularity is installed on LUMI and Tilde servers, so this step is omitted

# Need that installed (is in perkons already)
apt install apptainer-suid
 
# Make yourself fakeroot (use your userid)
sudo apptainer config fakeroot --add "ricards.rusevics@Tilde.lv"
```

### 1. Create directory

Make sure to use your directory (not /local_data/rusevich ...) here and in the rest of the tutorial.
```bash
export WORK=/local_data/rusevich/llm/skyrl-singularity
mkdir -p $WORK/{SkyRL,hf_cache,uv_cache,ray_tmp,home,harbor_data,harbor_trials,sif/tasks,apptainer_tmp/cache,harbor_trial_config}
git clone https://github.com/NovaSky-AI/SkyRL.git $WORK/SkyRL
```

### 2. Create the sif (container)

Use the skyrl.def file:
```bash
cd $WORK && singularity build --fakeroot sif/skyrl.sif skyrl.def
```

(it's ~8G btw)

### 3. Install datasets

```bash
#!/usr/bin/env bash
# 3_datasets.sh
set -euo pipefail
export WORK=/local_data/rusevich/llm/skyrl-singularity

singularity exec --bind "$WORK:$WORK" \
  --home "$WORK/home:$WORK/home" \
  --env HOME="$WORK/home" \
  --env HF_HOME="$WORK/hf_cache" \
  --env UV_CACHE_DIR="$WORK/uv_cache" \
  "$WORK/sif/skyrl.sif" bash -lc "
    cd $WORK/SkyRL

    # Train split: parquet of tar archives, extracted into task dirs
    uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
      --dataset open-thoughts/CodeContests \
      --output_dir $WORK/harbor_data/CodeContests

    # Eval split: already task dirs, the script only symlinks the HF snapshot
    uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
      --dataset open-thoughts/OpenThoughts-TB-dev \
      --output_dir $WORK/harbor_data/OpenThoughts-TB-dev
  "
```

(Optional) Check how many unique environments are in each dataset:
```bash
hash_envs() {
  root=$1; envname=${2:-environment}
  mapfile -d '' dirs < <(find -L "$root" -type d -name "$envname" -prune -print0 2>/dev/null)
  total=${#dirs[@]}
  [ "$total" -eq 0 ] && { echo "no '$envname' dirs under $root" >&2; return 1; }

  printf '%s\0' "${dirs[@]}" |
  xargs -0 -P "$(nproc)" -I{} sh -c '
    cd -P "$1" || exit
    find -L . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16
  ' _ {} |
  awk -v t="$total" '{printf "\r%d/%d", NR, t > "/dev/stderr"; print}
                     END {printf "\n" > "/dev/stderr"}' |
  sort | uniq -c | sort -rn
}

hash_envs "$WORK/harbor_data/CodeContests"
hash_envs "$WORK/harbor_data/OpenThoughts-TB-dev
```
In this tutorial I use CodeContests + OpenThoughts-TB-dev, modify for your datasets.
Each unique container will need to be created manually. How to do this for LUMI + large number of datasets - we'll take care of that later.


### 4. Copy blobs (OpenThoughts specifics)

```bash
export WORK=/local_data/rusevich/llm/skyrl-singularity
SRC=$WORK/harbor_data/OpenThoughts-TB-dev
DST=$WORK/harbor_data/eval-tb-dev

[ -d "$DST" ] || cp -rL "$SRC" "$DST"
find "$DST" -type f -exec chmod u+w {} +
ls "$DST" | wc -l
```

OpenThoughts has an interesting format, it's not possible to modify tasks' task.toml directly because of that.

### 5. Patch images (CodeContests)
CodeContest has one image for each task, because of that we can use the same container for each task.

Use the patch_images.py python script (MAKE SURE TO COPY IT FIRST), launch it like this:
```bash
DS=$WORK/harbor_data/CodeContests
IMG_DIR=$WORK/sif/images
mkdir -p "$IMG_DIR" "$WORK/tmp"

ENV_DIR=$(find "$DS" -mindepth 2 -maxdepth 2 -type d -name environment | head -n1)
docker build -t harbor-env:codecontests "$ENV_DIR"
docker save harbor-env:codecontests -o "$WORK/tmp/cc.tar"
singularity build --force "$IMG_DIR/codecontests.sif" "docker-archive://$WORK/tmp/cc.tar"
rm -f "$WORK/tmp/cc.tar"

python3 patch_images.py --dataset "$DS" --image "$IMG_DIR/codecontests.sif"
```


### 6. Patch images (OpenThoughts)

In OpenThoughts-TB-dev each task has unique environment. For space-economy reasons I'm gonna only use 10 tasks. <br>

Reducing image count:
```bash
SRC=$WORK/harbor_data/eval-tb-dev
EVAL=$WORK/harbor_data/eval-small
mkdir -p "$EVAL"
find "$SRC" -mindepth 1 -maxdepth 1 -type d | sort | head -n 10 |
  while read -r d; do cp -rL "$d" "$EVAL/"; done
```

Building + patching:
```bash
#!/usr/bin/env bash
# build_eval_images.sh [dataset-dir]
set -euo pipefail

DS=${1:-$WORK/harbor_data/eval-small}
IMG_DIR=$WORK/sif/images
TMP=$WORK/tmp
MANIFEST=$WORK/manifests/$(basename "$DS").tsv
mkdir -p "$IMG_DIR" "$TMP" "$(dirname "$MANIFEST")"
: > "$MANIFEST"

# private cache/tmp, the shared OCI cache is what corrupted the builds
export APPTAINER_CACHEDIR=$TMP/cache SINGULARITY_CACHEDIR=$TMP/cache
export APPTAINER_TMPDIR=$TMP/build SINGULARITY_TMPDIR=$TMP/build
mkdir -p "$TMP/cache" "$TMP/build"

mapfile -t TASKS < <(find "$DS" -mindepth 1 -maxdepth 1 -type d | sort)
N=${#TASKS[@]}
FAILED=0
echo "dataset: $DS ($N tasks)"

for i in "${!TASKS[@]}"; do
    task=${TASKS[$i]}
    name=$(basename "$task")
    sif=$IMG_DIR/$name.sif
    [ -f "$task/environment/Dockerfile" ] || { echo "[$((i+1))/$N] no Dockerfile: $name"; continue; }

    if [ -f "$sif" ]; then
        echo "[$((i+1))/$N] skip $name"
    else
        echo "[$((i+1))/$N] $(date +%T) building $name"
        if docker build -t "harbor-env:$name" "$task/environment" &&
           docker save "harbor-env:$name" -o "$TMP/$name.tar" &&
           singularity build --force --disable-cache "$sif" "docker-archive://$TMP/$name.tar"
        then
            echo "[$((i+1))/$N] $(date +%T) ok $name"
        else
            rm -f "$sif"
            FAILED=$((FAILED+1))
            echo "[$((i+1))/$N] $(date +%T) FAILED $name"
            rm -f "$TMP/$name.tar"
            continue
        fi
        rm -f "$TMP/$name.tar"
    fi
    printf '%s\t%s\n' "$name" "$sif" >> "$MANIFEST"
done

echo "built $(wc -l < "$MANIFEST")/$N, failed $FAILED -> $MANIFEST"
[ "$FAILED" -eq 0 ] || exit 1

python3 patch_images.py --dataset "$DS" --manifest "$MANIFEST"
```

### 7. Modify the launch script and run

```bash
# Make sure to change variables, especially available gpus
chmod +x run_harbor_singularity.sh
./run_harbor_singularity.sh
```