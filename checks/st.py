"""Read (a sample of) tensors straight from safetensors files with numpy: no torch, no mmap, little RAM."""
import json
import os
import struct

import numpy as np


class Model:
    """A HF model directory (sharded or single-file safetensors)."""

    def __init__(self, d):
        self.d = d
        idx = os.path.join(d, "model.safetensors.index.json")
        self._h = {}
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            self.map = {k: "model.safetensors" for k in self._header("model.safetensors")[0] if k != "__metadata__"}

    def _header(self, fn):
        if fn not in self._h:
            with open(os.path.join(self.d, fn), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._h[fn] = (json.loads(fh.read(n)), 8 + n)
        return self._h[fn]

    def info(self, key):
        return self._header(self.map[key])[0][key]

    def read(self, key, n):
        """First n elements of a tensor as float64, and its stored dtype ("F32" / "BF16")."""
        fn = self.map[key]
        hdr, base = self._header(fn)
        info = hdr[key]
        a, b = info["data_offsets"]
        size = {"F32": 4, "BF16": 2, "F16": 2}[info["dtype"]]
        with open(os.path.join(self.d, fn), "rb") as fh:
            fh.seek(base + a)
            buf = fh.read(min(b - a, n * size))
        if info["dtype"] == "F32":
            return np.frombuffer(buf, dtype="<f4").astype(np.float64), "F32"
        if info["dtype"] == "F16":
            return np.frombuffer(buf, dtype="<f2").astype(np.float64), "F16"
        return bf16_bits_to_float(np.frombuffer(buf, dtype="<u2")), "BF16"


def bf16_bits_to_float(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def round_bf16(x):
    """float -> nearest bf16 value, ties to even (what torch's .to(bfloat16) does)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).astype(np.float64)


def bf16_step(x):
    """Distance between neighbouring bf16 values around |x| (8 significant bits)."""
    x = np.maximum(np.abs(np.asarray(x, dtype=np.float64)), 1e-30)
    return 2.0 ** (np.floor(np.log2(x)) - 7)


def write_safetensors(path, tensors):
    """tensors: {name: (np array, "F32" | "BF16")}. For tests; BF16 arrays are rounded then stored as 16 bits."""
    header, blobs, off = {}, [], 0
    for name, (arr, dtype) in tensors.items():
        if dtype == "F32":
            raw = np.asarray(arr, dtype="<f4").tobytes()
        else:
            raw = (np.asarray(round_bf16(arr), dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
        header[name] = {"dtype": dtype, "shape": list(np.shape(arr)), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
