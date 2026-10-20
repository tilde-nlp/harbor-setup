"""sitecustomize for $ROOT/pyhooks (on PYTHONPATH via skyrl_env.sh).

Python imports `sitecustomize` once at startup from sys.path, and PYTHONPATH entries come
before site-packages, so this file runs in every process of the job: the driver, the Ray
workers, the vLLM engine/worker subprocesses. Keep it cheap: no torch, no vllm here.

If you already have a sitecustomize.py in pyhooks, merge the two try-blocks below into it.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import rccl_stream_hook

    rccl_stream_hook.install()
except Exception as _e:  # never break interpreter startup
    print(f"[sitecustomize] rccl_stream_hook not installed: {_e!r}", file=sys.stderr, flush=True)

# Chain to any other sitecustomize further down sys.path (e.g. one shipped in the venv),
# which this file would otherwise shadow.
try:
    import importlib.machinery
    import importlib.util

    _others = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
    _spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _others)
    if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
except Exception as _e:
    print(f"[sitecustomize] chained sitecustomize failed: {_e!r}", file=sys.stderr, flush=True)