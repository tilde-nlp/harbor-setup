#!/usr/bin/env bash
# Create a new working root (what a config's paths.root points at) from an existing, working one.
#
#   cd <new_root> && <repo>/setup.sh              new root = the current directory
#   <repo>/setup.sh <new_root> [<from_root>]
#   from_root defaults to the gsm8k/benchmark root below. Not covered (yet): Harbor (marin-harbor,
#   sif_cache, singularity_cache, tasks, trials) - copy those by hand for a Harbor root.
#
# Then point a config at it: paths.root: <new_root>, or per run: ./submit.sh <config> paths.root=<new_root>
# The new root must be under one of the config's paths.binds (e.g. /scratch/project_465003062).
#
# Safe to rerun: anything that already exists in <new_root> is left alone.
set -euo pipefail

FROM_DEFAULT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor

# Large and read-only during training: hard-linked (no extra space; survives deletion of the source).
# A symlink is made instead if a hard link is impossible (another file system or project quota).
LINK=(
  multitorch_sandbox.sif    # the container (paths.sandbox)
  tildeopen                 # the model (model.path); add other model dirs here
  hf                        # HF_HOME (paths.hf_home)
)
# Small, and may be rewritten per experiment: real copies, so the source can never be modified through them.
COPY=(
  data                      # paths.data: gsm8k parquets, pregen data/replay/...
)
# Outputs: created empty.
MKDIR=(logs ckpts exports previous_runs)

die() { echo "setup.sh: $*" >&2; exit 1; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
NEW=$(realpath -m "${1:-$PWD}")
FROM=$(realpath -m "${2:-$FROM_DEFAULT}")
case "$NEW/" in "$REPO"/*) die "new root $NEW is inside this repo; cd to the new root first (or pass it as an argument)" ;; esac
[ -d "$FROM" ] || die "source root $FROM does not exist"
[ "$NEW" != "$FROM" ] || die "new root and source root are the same"
mkdir -p "$NEW"
echo "new root: $NEW"
echo "from:     $FROM"

for item in "${LINK[@]}" "${COPY[@]}"; do
  [ -e "$FROM/$item" ] || die "$FROM/$item missing (edit the lists at the top of setup.sh)"
done

for item in "${LINK[@]}"; do
  if [ -e "$NEW/$item" ]; then echo "  exists    $item"; continue; fi
  if cp -al "$FROM/$item" "$NEW/$item" 2>/dev/null; then
    echo "  hardlink  $item"
  else
    rm -rf "${NEW:?}/$item"
    ln -s "$FROM/$item" "$NEW/$item"
    echo "  symlink   $item   (hard link not possible; deleting $FROM/$item breaks it)"
  fi
done

for item in "${COPY[@]}"; do
  if [ -e "$NEW/$item" ]; then echo "  exists    $item"; continue; fi
  cp -a "$FROM/$item" "$NEW/$item"
  echo "  copy      $item ($(du -sh "$NEW/$item" | cut -f1))"
done

for item in "${MKDIR[@]}"; do
  if [ -e "$NEW/$item" ]; then echo "  exists    $item/"; continue; fi
  mkdir -p "$NEW/$item"
  echo "  mkdir     $item/"
done

echo "done. Use it with:  paths.root: $NEW   (or ./submit.sh <config> paths.root=$NEW)"
