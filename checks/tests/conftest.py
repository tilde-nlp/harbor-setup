"""Shared helpers for the tests of the checks."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# Real inputs for the LUMI-only tests (test_known_bugs.py); every test that needs one is skipped without it.
MODEL_DIR = os.environ.get("CHECK_TEST_MODEL", os.path.join(ROOT, "tildeopen"))
BAD_EXPORT = os.environ.get("CHECK_TEST_BAD_EXPORT",
                            "/scratch/project_465003062/environment/containers/grpo-pregen-rl/exports/replay/global_step_162")
EXPORT_BASE = os.environ.get("CHECK_TEST_EXPORT_BASE", "/scratch/project_465003062/environment/containers/grpo-pregen-rl/tildeopen")


def variant(src, dst, edits):
    """
    Copy of a model dir: weight shards symlinked, every other file copied, then json edits applied.
    edits: {"file.json": fn(dict) -> None}
    """
    os.makedirs(dst)
    for fn in os.listdir(src):
        s = os.path.realpath(os.path.join(src, fn))
        if fn.endswith(".safetensors"):
            os.symlink(s, os.path.join(dst, fn))
        elif os.path.isfile(s) and not fn.endswith(".orig"):
            with open(s, "rb") as a, open(os.path.join(dst, fn), "wb") as b:
                b.write(a.read())
    for fn, edit in edits.items():
        p = os.path.join(dst, fn)
        obj = json.load(open(p))
        edit(obj)
        json.dump(obj, open(p, "w"), indent=2, ensure_ascii=False)
    return str(dst)


@pytest.fixture
def make_variant(tmp_path):
    def _make(name, edits, src=MODEL_DIR, dir_name=None):
        # keep the model dir's name: the repo's golden file is found by it (checks/golden/<name>.jsonl)
        (tmp_path / name).mkdir()
        return variant(src, tmp_path / name / (dir_name or os.path.basename(src.rstrip("/"))), edits)
    return _make
