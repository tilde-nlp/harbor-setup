# Driver for multi_node_nccl_test.py. Started by ray symmetric-run on the head node,
# inside the container, with skyrl_env.sh already sourced (same shell as run_training.sh).
set -x

: "${NUM_RANKS:=3}"
: "${MASTER_PORT:=12355}"
: "${SCRIPTS:=$ROOT/scripts}"

TEST="$SCRIPTS/multi_node_nccl_test.py"     # original script from SkyRL commit 59d4dae
[ -f "$TEST" ] || { echo "FATAL: $TEST not found" >&2; exit 1; }

ray status

# Record which settings the driver process has; the actors inherit the raylet
# environment instead, which the ENV lines of the RCCL logs will show.
env | grep -E "^(NCCL|RCCL|FI_|OFI_|CXI|HIP|SLINGSHOT)" | sort

python "$TEST" --num-nodes "$NUM_RANKS" --master-port "$MASTER_PORT"
rc=$?
echo "multi_node_nccl_test.py exit code: $rc"
exit $rc