#!/usr/bin/env bash
# Turn this clone into a working root: put the container, HF cache and data next to the code,
# taken from an existing, working root. The configs use the clone as paths.root (${derived.repo_dir}).
#
# The model directory is NOT set up here: which checkpoint to train is a per-experiment decision.
# Create <root>/tildeopen (model.path) by hand: symlink the weight shards of the checkpoint you want,
# copy (not link) its json / tokenizer files, then prepare it as described in checks/README.md
# ("Preparing a model directory") and run ./check.sh pre.
#
#   mkdir <dir> && cd <dir> && git clone <repo url> . && ./setup.sh [<from_root>]
#   from_root defaults to the gsm8k/benchmark root below; for pregen use .../grpo-pregen-rl (its data/replay).
#   Not covered (yet): Harbor (marin-harbor, sif_cache, singularity_cache, tasks, trials) - copy by hand.
#
# The clone must be under one of the configs' paths.binds (e.g. /scratch/project_465003062).
# Everything this creates is git-ignored (.gitignore is a whitelist of the repo's own files).
# Safe to rerun: anything that already exists is left alone.
set -euo pipefail

FROM_DEFAULT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor

# Large and not edited by hand: hard-linked (no extra space; survives deletion of the source).
# A symlink is made instead if a hard link is impossible (another file system or project quota).
LINK=(
  multitorch_sandbox.sif    # the container (paths.sandbox)
  hf                        # HF_HOME (paths.hf_home): hub models and datasets, ~150 GB
)
# Small, and may be rewritten per experiment: real copies, so the source can never be modified through them.
COPY=(
  data                      # paths.data: gsm8k parquets, pregen data/replay/...
)
# Outputs: created empty.
MKDIR=(logs ckpts exports previous_runs)

die() { echo "setup.sh: $*" >&2; exit 1; }
NEW=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)    # this clone
FROM=$(realpath -m "${1:-$FROM_DEFAULT}")
[ -d "$FROM" ] || die "source root $FROM does not exist"
[ "$NEW" != "$FROM" ] || die "this clone is the source root"
echo "root: $NEW"
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

[ -d "$NEW/tildeopen" ] || echo "note: no model directory yet. Create $NEW/tildeopen by hand (see the header of this file)"
echo "done. Check, then submit from here:"
echo "  ./check.sh pre configs/<task>/<setup>.yaml      (a new model dir also needs: ./check.sh gpu ...)"
echo "  ./submit.sh configs/<task>/<setup>.yaml"
