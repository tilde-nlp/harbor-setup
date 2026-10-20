"""rccl_stream_hook: keep vLLM's PyNcclCommunicator warm-up off the HIP null stream.

Problem (LUMI-G, ROCm 7.0.2, RCCL 2.26.6, aws-ofi-rccl 1.20.0 on libfabric 2.1/cxi):
an RCCL collective never completes when a rank that sends over the Libfabric plugin
launched it on the HIP null stream (handle 0). Launching on any user stream works, the
Socket transport works either way, GDR on or off makes no difference
(jobs 22831146 and 22831580, rccl_stream_bug.py).

Where SkyRL hits it (commit 59d4dae, vLLM 0.22.1): the weight-sync group is built on both
sides by vllm.distributed.weight_transfer.nccl_engine.NCCLWeightTransferEngine
._stateless_init_process_group -> PyNcclCommunicator(pg, device). The constructor runs a
one-element warm-up all_reduce on vllm.utils.torch_utils.current_stream(), which returns
the stream last recorded by vLLM's patched torch.cuda.set_stream in the calling thread.
After any `with torch.cuda.stream(...)` block has exited in that thread, that is the null
stream, and the trainer rank (alone on its node, hence a network sender) hangs in the
warm-up. The actual weight broadcasts (packed_nccl_broadcast_producer/consumer) already
run inside `with torch.cuda.stream(torch.cuda.Stream())` and are not affected.

What this hook does: wraps PyNcclCommunicator.__init__ so that, when the stream vLLM would
use is the null stream, the constructor runs inside `with torch.cuda.stream(s)` for a fresh
stream s on the communicator's device. The context manager restores the previous stream
on exit. Nothing else changes: transport settings, GDR, the broadcast path and every other
communicator (vLLM's TP communicator, FSDP's process groups) are untouched.

Installation: the hook is installed lazily through a sys.meta_path finder, so importing this
module costs nothing until vllm.distributed.device_communicators.pynccl is imported.
sitecustomize.py next to this file calls install() in every Python process that has
$ROOT/pyhooks on PYTHONPATH (trainer workers, vLLM engine and worker processes, driver).

Environment:
  RCCL_STREAM_HOOK=0       disable the hook entirely
  RCCL_STREAM_HOOK=force   build every PyNcclCommunicator under a fresh user stream, whatever
                           stream vLLM has recorded (diagnostic)
  (default)                act only when the recorded stream is the null stream

Every construction prints one line with the rank, device, thread and the recorded stream, so
the per-process logs show which stream the warm-up all_reduce was launched on.
"""
from __future__ import annotations

import importlib.abc
import os
import sys
import threading

TARGET_MODULE = "vllm.distributed.device_communicators.pynccl"
_TAG = "[rccl_stream_hook]"
_lock = threading.Lock()
_installed = False


def _say(msg: str) -> None:
    print(f"{_TAG} {msg}", file=sys.stderr, flush=True)


def _patch_pynccl(mod) -> None:
    """Wrap PyNcclCommunicator.__init__ in `mod` (idempotent)."""
    import torch
    from vllm.utils import torch_utils as _tu

    cls = getattr(mod, "PyNcclCommunicator", None)
    if cls is None or getattr(cls, "_rccl_stream_hook_installed", False):
        return
    orig_init = cls.__init__

    def __init__(self, group, device, *args, **kwargs):
        if isinstance(device, int):
            dev = torch.device("cuda", device)
        else:
            dev = torch.device(device)
        recorded = getattr(_tu._current_stream_tls, "value", None)
        if isinstance(recorded, torch.cuda.Stream):
            seen = f"recorded stream {recorded.cuda_stream:#x} (device {recorded.device})"
            on_null = recorded.cuda_stream == 0
        else:
            seen = f"nothing recorded ({type(recorded).__name__}); vLLM will create a user stream"
            on_null = False
        rank = getattr(group, "rank", "?")
        world = getattr(group, "world_size", "?")
        tname = threading.current_thread().name
        force = os.environ.get("RCCL_STREAM_HOOK", "1").lower() == "force"
        if not (on_null or force):
            # Either a user stream is already current, or nothing is recorded yet and vLLM's
            # current_stream() will create and set a dedicated stream itself.
            _say(f"pid {os.getpid()} rank {rank}/{world} device {dev} thread {tname}: {seen}; no action")
            return orig_init(self, group, device, *args, **kwargs)
        s = torch.cuda.Stream(device=dev)
        _say(
            f"pid {os.getpid()} rank {rank}/{world} device {dev} thread {tname}: {seen}; "
            f"{'forced' if force else 'null stream'}: building PyNcclCommunicator on stream {s.cuda_stream:#x}"
        )
        with torch.cuda.stream(s):
            orig_init(self, group, device, *args, **kwargs)
        # keep the stream alive for the communicator's lifetime (harmless, pool-backed)
        self._rccl_stream_hook_stream = s

    __init__.__wrapped__ = orig_init
    cls.__init__ = __init__
    cls._rccl_stream_hook_installed = True
    _say(f"pid {os.getpid()}: PyNcclCommunicator.__init__ wrapped")


class _LoaderWrapper(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch_pynccl(module)
        except Exception as e:  # never break the import because of the hook
            _say(f"patch failed, hook inactive: {e!r}")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _LoaderWrapper(spec.loader)
                return spec
        return None


def install() -> bool:
    """Install the lazy patch. Returns True when active (or already active)."""
    global _installed
    if os.environ.get("RCCL_STREAM_HOOK", "1") == "0":
        return False
    with _lock:
        if _installed:
            return True
        if TARGET_MODULE in sys.modules:
            _patch_pynccl(sys.modules[TARGET_MODULE])
        else:
            sys.meta_path.insert(0, _PatchOnImport())
        _installed = True
        return True