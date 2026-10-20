"""
sitecustomize.py  --  drop into $ROOT/pyhooks (already on PYTHONPATH in apply_env).

Enables torch CUDA/HIP memory history in every Python process of the job and dumps
memory_viz-compatible snapshot pickles. No SkyRL source changes.

Activation (all read from the environment, so pass them via EXTRA_ENV):
  SKYRL_MEMPROF=1                 master switch; without it this file does nothing.
  SKYRL_MEMPROF_DIR=<dir>         where pickles go. Default: $LOG_DIR/memsnap-$SLURM_JOB_ID
  SKYRL_MEMPROF_STACKS=all|python stack capture. "all" adds C++ frames, costs more. Default: python
  SKYRL_MEMPROF_MAX_ENTRIES=N     ring buffer of allocation events kept per process. Default: 200000
  SKYRL_MEMPROF_INTERVAL=S        periodic dump every S seconds, 0 disables. Default: 0
  SKYRL_MEMPROF_MATCH=<substr>    only record in processes whose argv contains this substring,
                                  e.g. "ray::" for Ray actors. Default: record everywhere.

Snapshots are written:
  - on OOM (file suffix .oom.pickle), via the allocator's out-of-memory observer
  - on SIGTERM / SIGINT / normal exit, best effort (Ray kills idle workers with SIGKILL eventually)
  - every SKYRL_MEMPROF_INTERVAL seconds, overwriting the same file
  - on demand: `touch $SKYRL_MEMPROF_DIR/DUMP_NOW` from a login node. Every process polls the
    file's mtime every 10 s and dumps once per touch.

Open the pickles at https://pytorch.org/memory_viz (drag and drop), or convert with
  python -m torch.cuda._memory_viz trace_plot x.pickle -o x.html

Recording starts at the first CUDA/HIP initialisation of the process (hook on
torch.cuda._lazy_init), never at import time, so processes that fork after importing torch
are unaffected.
"""

import os
import sys

if os.environ.get("SKYRL_MEMPROF") == "1":
    import importlib.abc
    import importlib.machinery

    _match = os.environ.get("SKYRL_MEMPROF_MATCH", "")

    def _wanted():
        if not _match:
            return True
        try:
            import setproctitle  # Ray renames workers to "ray::ActorName"; argv does not change
            title = setproctitle.getproctitle()
        except Exception:
            title = ""
        return _match in " ".join(sys.argv) or _match in title

    class _TorchCudaHook(importlib.abc.MetaPathFinder):
        """Post-import hook: after torch.cuda is executed, wrap torch.cuda._lazy_init."""

        def find_spec(self, name, path, target=None):
            if name != "torch.cuda":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            sys.meta_path.remove(self)
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _install(module)
                except Exception as e:  # never break the job because of the profiler
                    sys.stderr.write(f"[memprof] install failed: {e!r}\n")

            spec.loader.exec_module = exec_module
            return spec

    def _install(cuda):
        orig_lazy_init = cuda._lazy_init
        state = {"started": False}

        def lazy_init():
            orig_lazy_init()
            if not state["started"]:
                state["started"] = True  # set before calling into torch, which re-enters _lazy_init
                if _wanted():
                    _start_recording()

        cuda._lazy_init = lazy_init

    def _start_recording():
        import atexit
        import signal
        import socket
        import threading
        import time

        import torch

        out_dir = os.environ.get("SKYRL_MEMPROF_DIR") or os.path.join(
            os.environ.get("LOG_DIR", "."), f"memsnap-{os.environ.get('SLURM_JOB_ID', 'x')}"
        )
        os.makedirs(out_dir, exist_ok=True)
        stacks = os.environ.get("SKYRL_MEMPROF_STACKS", "python")
        max_entries = int(os.environ.get("SKYRL_MEMPROF_MAX_ENTRIES", "200000"))
        interval = float(os.environ.get("SKYRL_MEMPROF_INTERVAL", "0"))
        host = socket.gethostname().split(".")[0]
        pid = os.getpid()
        hip = os.environ.get("HIP_VISIBLE_DEVICES", "")

        def _rank():
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    return f"rank{dist.get_rank()}"
            except Exception:
                pass
            return "norank"

        def _base():
            # rank is resolved at dump time; torch.distributed is not initialised yet at CUDA init
            return os.path.join(out_dir, f"{host}.{pid}.{_rank()}")

        lock = threading.Lock()

        def dump(tag):
            with lock:
                path = f"{_base()}.{tag}.pickle"
                try:
                    torch.cuda.memory._dump_snapshot(path)
                    _log(f"dumped {path}")
                except Exception as e:
                    _log(f"dump {path} failed: {e!r}")

        def _log(msg):
            line = f"{time.strftime('%H:%M:%S')} {host} pid={pid} hip={hip} {msg}\n"
            try:
                with open(os.path.join(out_dir, "memprof.log"), "a") as f:
                    f.write(line)
            except Exception:
                sys.stderr.write("[memprof] " + line)

        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks=stacks, max_entries=max_entries
        )
        _log(f"recording started stacks={stacks} max_entries={max_entries} argv={' '.join(sys.argv)[:200]}")

        # 1. OOM: dump the state at the moment the allocation failed
        def oom_observer(device, alloc, device_alloc, device_free):
            _log(f"OOM device={device} alloc={alloc} device_alloc={device_alloc} device_free={device_free}")
            dump("oom")

        try:
            torch._C._cuda_attach_out_of_memory_observer(oom_observer)
        except Exception as e:
            _log(f"no OOM observer: {e!r}")

        # 2. exit paths, best effort
        atexit.register(dump, "exit")

        def on_signal(signum, frame):
            dump(f"sig{signum}")
            signal.signal(signum, signal.SIG_DFL)
            os.kill(pid, signum)

        if threading.current_thread() is threading.main_thread():
            for s in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal.signal(s, on_signal)
                except Exception:
                    pass

        # 3. periodic and on-demand (DUMP_NOW trigger file)
        trigger = os.path.join(out_dir, "DUMP_NOW")

        def poller():
            last_trigger = 0.0
            last_periodic = time.time()
            while True:
                time.sleep(10)
                try:
                    m = os.stat(trigger).st_mtime
                except OSError:
                    m = 0.0
                if m > last_trigger:
                    last_trigger = m
                    dump("ondemand")
                if interval > 0 and time.time() - last_periodic >= interval:
                    last_periodic = time.time()
                    dump("periodic")

        threading.Thread(target=poller, name="memprof-poller", daemon=True).start()

    sys.meta_path.insert(0, _TorchCudaHook())