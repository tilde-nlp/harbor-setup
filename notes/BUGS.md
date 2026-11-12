


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


## Run hangs in the first forward pass on nid[005018-005021]

This is on 12.11.26, dev-g.

### What happens

The job reaches step 1, logs `Started: 'fwd_logprobs_values_reward'` and never logs anything again. No error, no
OOM, host RAM flat on every node. Stacks (`fsdp-skyrl-harbor/inspect_hung_job.sh <jobid>`): all 32 reference
workers sit in the first attention layer, in `bool()` of a GPU tensor (`modeling_flash_attention_utils.py:773`),
i.e. in a GPU sync that waits for queued GPU work (the collectives before it) that never finishes. The policy
workers are idle, the driver waits for the reference.

### Evidence that it is the nodes

| jobs | nodes | first forward pass |
|---|---|---|
| 23677895, 23681198 | nid[005018-005021] | hung, both |
| 23674901, 23675865, 23676681, 23677124 | nid[005002-005014] | ~40 s |
| 23678761 (same code as the hang before it), 23682492 | nid[007961-007964] | ~40 s |

Two hangs, both on this node set, eight passes on other nodes, with the same code on both sides. Which of the
four nodes (or which GPU / link) is at fault is not known: the stacks are the same on all four. One bad
GPU or link stalls everyone, because the model is sharded over all 32 GCDs and every forward pass runs collectives
across all nodes. Not reported to LUMI support yet.

### What to do

- The pregen configs exclude the nodes: `slurm.extra: ["--exclude=nid[005018-005021]"]`.
- `launch/job.sbatch` has a watchdog: once training has started, 15 min without a new line in the Slurm log
  cancels the job and writes `<run dir>/HUNG` with the node list and the exclude override (`HANG_TIMEOUT_MIN`).
  Then resubmit with `skyrl.trainer.resume_mode=latest`. A hang before the first step is not covered.
- If it happens on other nodes: rerun once with `env.NCCL_DEBUG=INFO` (logs every collective) and add the nodes here.


## Host OOM in the HF export when the reference shares the policy GPUs

Jobs 23674901, 23675865 (and grpo-pregen-rl 23282545): `oom_kill` on the rank-0 node in `save_hf_model` with
~290 GB of the node free.

### Explanation

SkyRL's `Worker.offload_to_cpu()` calls `_set_numa_affinity()`: `numa_set_membind` on the NUMA node of the worker's
GPU. That binding is strict, and a LUMI-G node has 4 NUMA nodes of 126 GB. Workers only offload when policy and
reference are colocated (they swap between GPU and host RAM), so only then is the binding set. For the export rank 0
gathers the whole fp32 model (120 GB) in its own process: it does not fit in one NUMA node next to three other
workers, and the kernel kills it instead of using the other NUMA nodes. `previous_runs/<run>/mem/<node>.log`
(host RAM every 5 s) is what showed that the node was not full.

### Fix

`skyrl_extensions/pregenerated/entrypoint.py`, `SpillPolicyWorkerBase`: same CPU binding, memory policy back to
"local node first, spill when full". Export node peak afterwards: 225 GB of 480 GB (job 23682492).
Two more things found on the way: the staged model copy in /tmp costs 58 GB of RAM per node and is deleted once
the models are loaded (`free_staged_weights`); and the export failed in `import deepspeed` on a host module path
(`CMAKE_PREFIX_PATH`) that is unreadable in the container (`launch/lumi_env.sh`).

