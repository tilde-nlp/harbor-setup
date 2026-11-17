#!/usr/bin/env bash
# Convert an fp32 HF export (safetensors) to bf16, next to it: <dir> -> <dir>_bf16.
#
#   tools/to_bf16.sh exports/<run>/global_step_25/policy      -> exports/<run>/global_step_25/policy_bf16
#   tools/to_bf16.sh <dir> <out_dir>
#
# One shard at a time (peak RAM ~ one fp32 shard + its bf16 copy). Floating tensors become bf16, others are
# copied unchanged; shard names and the weight map stay the same, the index's total_size is recomputed,
# config.json gets torch_dtype/dtype = bfloat16, every other file (tokenizer, generation config) is copied.
# Runs on a login node (no GPU) with the container's Python, like ./check.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
SANDBOX="${SANDBOX:-$ROOT/multitorch_sandbox.sif}"
BINDS="${CHECK_BINDS:-/scratch/project_465003062,/pfs}"
[ -e "$SANDBOX" ] || { echo "to_bf16.sh: container $SANDBOX missing (a fresh clone needs ./setup.sh first)" >&2; exit 1; }
[ $# -ge 1 ] || { sed -n '2,5p' "$0"; exit 2; }
SRC=$(cd "$1" && pwd -P)
DST=${2:-${SRC%/}_bf16}
export PYTHONNOUSERSITE=1

singularity exec -B "$BINDS" "$SANDBOX" /opt/venv/bin/python - "$SRC" "$DST" <<'EOF'
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1:3]
if os.path.exists(dst):
    sys.exit(f"to_bf16.sh: {dst} already exists, remove it first")
shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
if not shards:
    sys.exit(f"to_bf16.sh: no .safetensors in {src}")
tmp = dst + ".partial"
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)

total, dtypes = 0, set()
for i, name in enumerate(shards, 1):
    out = {}
    with safe_open(os.path.join(src, name), framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            t = f.get_tensor(k)
            dtypes.add(str(t.dtype))
            if t.is_floating_point():
                t = t.to(torch.bfloat16)
            out[k] = t.contiguous()
            total += t.numel() * t.element_size()
    save_file(out, os.path.join(tmp, name), metadata={**meta, "format": "pt"})
    print(f"[{i}/{len(shards)}] {name}", flush=True)
    del out

for name in os.listdir(src):
    p = os.path.join(src, name)
    if name.endswith(".safetensors") or not os.path.isfile(p):
        continue
    if name == "config.json":
        cfg = json.load(open(p))
        def set_dtype(d):
            for key in ("torch_dtype", "dtype"):
                if key in d:
                    d[key] = "bfloat16"
            for v in d.values():
                if isinstance(v, dict):
                    set_dtype(v)
        cfg.setdefault("torch_dtype", "bfloat16")
        set_dtype(cfg)
        json.dump(cfg, open(os.path.join(tmp, name), "w"), indent=2)
    elif name == "model.safetensors.index.json":
        idx = json.load(open(p))
        idx.setdefault("metadata", {})["total_size"] = total
        json.dump(idx, open(os.path.join(tmp, name), "w"), indent=2)
    else:
        shutil.copy2(p, tmp)

os.rename(tmp, dst)
print(f"source dtypes: {', '.join(sorted(dtypes))}")
print(f"wrote {dst} ({total / 2**30:.1f} GiB of weights)")
EOF
