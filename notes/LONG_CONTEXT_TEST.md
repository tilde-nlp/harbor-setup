```bash

export ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor
export SANDBOX="$ROOT/multitorch_sandbox"
export MODEL="$ROOT/tildeopen"

mkdir -p "$ROOT/data/ctx64k"
cp -r "$ROOT/data/smoke/"* "$ROOT/data/ctx64k/"

singularity exec -B /pfs,/scratch,/projappl,/project,/flash,/appl,/users "$SANDBOX" \
  bash -c '. /opt/venv/bin/activate && export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 && python -' <<'EOF'
import os, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ["MODEL"])
TARGET = 55_000
filler = "The quick brown fox jumps over the lazy dog. "
n_filler = len(tok(filler, add_special_tokens=False).input_ids)
for f in sorted(glob.glob(os.environ["ROOT"] + "/data/ctx64k/*/instruction.md")):
    base = open(f).read()
    n = len(tok(base).input_ids)
    reps = max(0, (TARGET - n) // n_filler + 1)
    open(f, "w").write("# Reference material (ignore)\n\n" + filler * reps + "\n\n" + base)
    print(f, len(tok(open(f).read()).input_ids), "tokens")
EOF

```