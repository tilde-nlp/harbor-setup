"""
HF export of the policy for pregenerated runs: same result as SkyRL's FSDPStrategy.save_hf_model (fp32 safetensors
shards + index + config), written differently.

Why: SkyRL's export took ~200 s for TildeOpen-30B (job 23677124): ~45 s to gather the 120 GB state dict on rank 0,
then ~150 s in save_pretrained, which writes three 50 GB shards one after the other from one thread (0.8 GB/s on
Lustre). Here the shards are small (SHARD_BYTES) and written by WRITERS threads at once; the tensor bytes go to the
file straight from the tensor's memory (no serialized copy), and the write syscall releases the GIL.

Also different: the tokenizer files are copied from the start model instead of tokenizer.save_pretrained(). The
latter writes a converted tokenizer.json that drops the word-boundary marker after control tokens
(notes/TRAINING_SANITY_CHECK.md 4; `./check.sh export` -> export.tokenizer FAIL).

File format: https://github.com/huggingface/safetensors#format  (8-byte little-endian header size, JSON header
padded with spaces to a multiple of 8, then the tensors back to back in header order).
"""
import glob
import json
import os
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

# imported here, not inside save_hf_model: a wrong path must fail at job start, not at the first export
from skyrl.backends.skyrl_train.distributed.fsdp_utils import fsdp2_get_full_state_dict
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

SHARD_BYTES = 4 * 2**30   # a tensor larger than this gets a shard of its own
WRITERS = 16
TOKENIZER_FILES = ("tokenizer*", "*.model", "*.vocab", "special_tokens_map.json", "added_tokens.json", "chat_template*")

_DTYPE = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
}


def _nbytes(t):
    return t.numel() * t.element_size()


def plan_shards(state_dict, shard_bytes=SHARD_BYTES):
    """[[name, ...], ...]: tensors in state-dict order, a new shard when the next tensor would pass shard_bytes."""
    shards, size = [[]], 0
    for name, t in state_dict.items():
        if shards[-1] and size + _nbytes(t) > shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += _nbytes(t)
    return shards


def write_shard(path, tensors):
    """tensors: {name: CPU tensor}. Writes one .safetensors file."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, t in tensors.items():
        header[name] = {"dtype": _DTYPE[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + _nbytes(t)]}
        offset += _nbytes(t)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for t in tensors.values():
            if t.numel():
                # the tensor's own memory as bytes: no copy, works for every dtype (numpy has no bf16)
                f.write(memoryview(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))


def write_state_dict(state_dict, out_dir, shard_bytes=SHARD_BYTES, writers=WRITERS):
    """All shards in parallel + model.safetensors.index.json (the layout transformers writes and reads)."""
    for name, t in state_dict.items():
        if t.device.type != "cpu" or t.dtype not in _DTYPE:
            raise ValueError("cannot export {}: device {}, dtype {}".format(name, t.device, t.dtype))
    shards = plan_shards(state_dict, shard_bytes)
    names = ["model-{:05d}-of-{:05d}.safetensors".format(i + 1, len(shards)) for i in range(len(shards))]
    if len(shards) == 1:
        names = ["model.safetensors"]
    with ThreadPoolExecutor(max_workers=writers) as pool:
        jobs = [pool.submit(write_shard, os.path.join(out_dir, fn), {k: state_dict[k] for k in keys})
                for fn, keys in zip(names, shards)]
        for j in jobs:
            j.result()
    if len(shards) > 1:
        index = {
            "metadata": {"total_parameters": sum(t.numel() for t in state_dict.values()),
                         "total_size": sum(_nbytes(t) for t in state_dict.values())},
            "weight_map": {k: fn for fn, keys in zip(names, shards) for k in keys},
        }
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)
    return len(shards)


def copy_tokenizer_files(src_dir, out_dir):
    """The start model's tokenizer files, byte for byte. Returns the copied names ([] if src_dir has none)."""
    copied = []
    for pattern in TOKENIZER_FILES:
        for path in glob.glob(os.path.join(src_dir, pattern)):
            if os.path.isfile(path) and os.path.basename(path) not in copied:
                shutil.copy(path, out_dir)
                copied.append(os.path.basename(path))
    return copied


def save_hf_model(worker, export_dir, tokenizer):
    """Drop-in for Worker.save_hf_model on an FSDP policy worker. Collective: runs on every rank."""
    strategy = worker.strategy
    hf_model = strategy._unwrap_model(worker.model)
    fsdp_model = worker.model.model if isinstance(worker.model, HFModelWrapper) else worker.model
    t0 = time.time()
    state_dict = fsdp2_get_full_state_dict(fsdp_model, cpu_offload=True, rank0_only=True)   # all ranks take part
    if strategy.is_rank_0():
        t1 = time.time()
        with strategy._atomic_local_export_dir(export_dir) as work_dir:
            n = write_state_dict(state_dict, work_dir)
            strategy._fix_fsdp_config(hf_model.config).save_pretrained(work_dir)
            if getattr(hf_model, "generation_config", None) is not None and hf_model.can_generate():
                hf_model.generation_config.save_pretrained(work_dir)
            copied = copy_tokenizer_files(str(getattr(hf_model.config, "_name_or_path", "") or ""), work_dir)
            if not copied and tokenizer is not None:
                tokenizer.save_pretrained(work_dir)
        gb = sum(_nbytes(t) for t in state_dict.values()) / 2**30
        print("[rank-0]: HF export to {}: gather {:.0f} s, write {:.0f} s ({:.0f} GB, {} tensors, {} shards), tokenizer {}".format(
            export_dir, t1 - t0, time.time() - t1, gb, len(state_dict), n,
            "copied from the start model: " + " ".join(sorted(copied)) if copied else "saved by transformers (start model dir not found)"),
            flush=True)
    del state_dict
    dist.barrier()
