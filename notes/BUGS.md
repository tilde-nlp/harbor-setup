


## One gcd is not found (7/8)

This is on 08.09.26, slurm 25.05.9

### Repro

1-node (expect: both lines show all 8):

```bash
sbatch -p dev-g -A project_465003062 -N1 --ntasks-per-node=1 --gpus-per-node=8 \
  -c 56 --mem=0 --exclusive -t 0:02:00 -o repro-1node-%j.out --wrap '
echo plain:;   srun -N1 -n1 printenv ROCR_VISIBLE_DEVICES
echo overlap:; srun -N1 -n1 --overlap printenv ROCR_VISIBLE_DEVICES'
```

2-node (expect: plain 8, overlap 7):

```bash
sbatch -p dev-g -A project_465003062 -N2 --ntasks-per-node=1 --gpus-per-node=8 \
  -c 56 --mem=0 --exclusive -t 0:02:00 -o repro-2node-%j.out --wrap '
echo plain:;   srun -N1 -n1 printenv ROCR_VISIBLE_DEVICES
echo overlap:; srun -N1 -n1 --overlap printenv ROCR_VISIBLE_DEVICES'
```

### Explanation
For multi-node setup (this is important, because for single node there is no such issue) --overlap flag will fuck everything up. Slurm 26 fixes this, which lumi doesn't have at the time of writing this


## "No available shared memory broadcast block found in 60 seconds" timeout

Full text: <br>
(EngineCore pid=120443) INFO 10-04 14:12:30 [shm_broadcast.py:698] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).

### Repro

Idfk, but this happens when I use colocate_all=false IN SOME CASES, but in some it doesn't. Basically weight_sync (or other step like that) is stuck for 10 minutes and this triggers a timeout in skyrl. <br>

### Fix

```bash
# Weight sync is one RCCL communicator of 1 trainer GCD + 8 GCDs per vLLM node. RCCL's Rome
# topology-model matching is a per-process decision (needs all 8 GCDs of the node in the comm):
# the 8-GCD ranks match MI250X model 36 and build an extra ring connection set, the 1-GCD rank
# does not, and ncclCommInitRank deadlocks after "Connected all rings". Reproduced in
# scripts/rccl_hetero_test.sbatch (jobs 22551317 hang / 22551365 ok); training jobs 22231718,
# 22429724, 22494530, 22526638, 22546678, 22549832 all died this way. Must be set on every rank.
export RCCL_MODEL_MATCHING_DISABLE=1
```

