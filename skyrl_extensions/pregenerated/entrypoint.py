"""
Entrypoint: GRPO on pre-generated rollouts with SkyRL.

Everything algorithmic is a SkyRL built-in, selected in configs/pregen/grpo_bs5.yaml:
    trainer.algorithm.policy_loss_type=rollout_is      banded importance ratio, stop-gradient weight,
                                                       zero outside (1-eps_clip_low, 1+eps_clip_high)
    trainer.algorithm.use_kl_loss=true                 k3 KL to the reference worker (frozen start checkpoint)

What is custom here:
    PregeneratedGenerator / PregeneratedDataset   stored rollouts instead of sampling (generator.py)
    PregeneratedTrainer.train_critic_and_policy
                                      copies the reference worker's log-probs (base_action_log_probs,
                                      same FSDP/Ulysses stack as the policy) into rollout_logprobs, so the
                                      rollout_is ratio is pi_theta / pi_ref, exactly 1 at step 1
    NoInferenceClient + PregeneratedTrainer.init_weight_sync_state
                                      no vLLM engines are started and no weights are pushed: the
                                      inference node and the per-step broadcast are gone
    SpillPolicyWorker                 SkyRL binds a worker's memory strictly to its GPU's NUMA node (126 GB) the
                                      first time it offloads, i.e. only when the reference shares the policy
                                      GPUs. Rank 0 then cannot gather the 120 GB state dict for the HF export
                                      (host OOM with 290 GB free). Keeps the CPU binding, lets memory spill over.
                                      Its HF export is export.py: shards written in parallel, tokenizer files
                                      copied from the start model
    free_staged_weights               once policy and reference are loaded, the weight files of the staged
                                      model copy in /tmp (node RAM, ~60 GB per node) are deleted

Run (inside the container, under ray):
    python -m skyrl_extensions.pregenerated.entrypoint data.train_data="['/path/pregenerated.parquet']" <skyrl overrides>
"""
import glob
import os
import sys
from ctypes import CDLL
from ctypes.util import find_library

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from skyrl_extensions.pregenerated import export
from skyrl_extensions.pregenerated.generator import PregeneratedDataset, PregeneratedGenerator, PregeneratedStore

class NoInferenceClient:
    """
    Stands in for the vLLM engines, which are never started. Every method is an awaitable no-op:
    the trainer only calls sleep()/wake_up()/pause/resume on the client (trainer.py 381, 573), the
    generator never touches it, and the weight-sync group is skipped below. If a traceback shows
    the trainer reading a value from this object, add that one method explicitly.
    Private and dunder names are not answered: copy/pickle/hasattr probes must see a normal object.
    """

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _noop(*args, **kwargs):
            return None
        return _noop


class SpillPolicyWorkerBase(FSDPPolicyWorkerBase):
    """
    worker.py offload_to_cpu() -> _set_numa_affinity(): numa_run_on_node_mask + numa_set_membind on the NUMA node
    of the worker's GPU. membind is strict: when that node (126 GB on LUMI-G, shared by 2 policy and 2 reference
    workers) is full the kernel OOM-kills instead of using the other three nodes. Training never needs that much
    per worker, the HF export does: rank 0 gathers the whole fp32 model (120 GB) in host RAM. Jobs 23674901 and
    23675865 died there with ~290 GB of the node free (previous_runs/<run>/mem/). Without colocation nothing is
    ever offloaded, so the binding is never set and the export works.
    Here: same CPU binding, then back to the default policy (allocate on the local node, spill when it is full).
    """

    def _set_numa_affinity(self, rank):
        super()._set_numa_affinity(rank)
        try:
            CDLL(find_library("numa")).numa_set_localalloc()
        except Exception as e:  # no libnuma: the parent did not bind either
            print("SpillPolicyWorker: numa_set_localalloc not applied: {}".format(e), flush=True)

    def save_hf_model(self, export_dir, tokenizer):
        # parallel shard writer + the start model's tokenizer files: skyrl_extensions/pregenerated/export.py
        export.save_hf_model(self, export_dir, tokenizer)


SpillPolicyWorker = ray.remote(num_gpus=1)(SpillPolicyWorkerBase)


@ray.remote(num_cpus=0)
def _remove_weight_files(model_dir):
    freed = 0
    for path in glob.glob(os.path.join(model_dir, "*.safetensors")) + glob.glob(os.path.join(model_dir, "*.bin")):
        freed += os.path.getsize(path)
        os.remove(path)
    return freed


def free_staged_weights(model_dir):
    """
    launch/job.sbatch copies the model to /tmp on every node. /tmp is RAM charged to the job, and after
    the workers have loaded policy and reference nothing reads the weight files again (no vLLM here; the HF
    export writes config and tokenizer from memory, a resume reads ckpts/). With the reference on the policy
    GPUs the HF export ran out of host RAM on the rank-0 node (job 23674901): this gives 60 GB back per node.
    The small files (config, tokenizer) stay. Only touches a copy under /tmp.
    """
    if not os.path.realpath(model_dir).startswith("/tmp/"):
        return
    nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"]]
    freed = ray.get([
        _remove_weight_files.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n, soft=False)).remote(model_dir)
        for n in nodes
    ])
    print("free_staged_weights: removed {:.0f} GB of weight files from {} on {} nodes".format(sum(freed) / 2**30, model_dir, len(nodes)), flush=True)


class PregeneratedTrainer(RayPPOTrainer):
    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        if self.cfg.trainer.strategy == "fsdp":
            PolicyWorker = SpillPolicyWorker
        return super().build_models(PolicyWorker, CriticWorker, RefWorker)

    def init_weight_sync_state(self):
        # trainer.py 855: would build the NCCL group between policy workers and engines. Instead,
        # disable the two pushes that follow it (initial sync at 289, per-step sync at 494), which
        # both call self.dispatch.save_weights_for_sampler(). Runs after self.dispatch exists.
        async def _no_sync(model_id=None):
            return None
        self.dispatch.save_weights_for_sampler = _no_sync
        # Called from the trainer right after build_models: policy and reference are loaded on every worker.
        free_staged_weights(self.cfg.trainer.policy.model.path)

    def train_critic_and_policy(self, data):
        base = data.get("base_action_log_probs", None)
        if base is None:
            raise ValueError("rollout_is needs the reference worker's log-probs: set trainer.algorithm.use_kl_loss=true")
        data["rollout_logprobs"] = base
        return super().train_critic_and_policy(data)


class PregeneratedExp(BasePPOExp):
    def get_inference_client(self):
        # main_base.py 210: would start the vLLM servers (an extra node for a 30B model). Nothing
        # here generates, so no engine, no placement group, no weight sync.
        return NoInferenceClient()

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        alg = cfg.trainer.algorithm
        assert alg.policy_loss_type == "rollout_is" and alg.use_kl_loss, (
            "pregenerated training needs trainer.algorithm.policy_loss_type=rollout_is and use_kl_loss=true"
        )
        return PregeneratedGenerator(PregeneratedStore(cfg.data.train_data[0], max_seq_len=alg.max_seq_len))

    def get_train_dataset(self):
        return PregeneratedDataset(self.cfg.data.train_data[0], self.cfg.generator.n_samples_per_prompt)

    def get_eval_dataset(self):
        return None

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
        return PregeneratedTrainer(
            cfg=cfg, tracker=tracker, tokenizer=tokenizer, train_dataset=train_dataset,
            eval_dataset=eval_dataset, inference_engine_client=inference_engine_client,
            generator=generator, colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    PregeneratedExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must cover the longest prompt + response in the parquet")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()