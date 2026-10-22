"""Carga de pesos sem 2a cópia anônima.

laya.agent usa safetensors.load_file (copia 800 MB para RAM anônima) depois de já ter
construído o modelo fp32 (outros 800 MB): pico ~1,8 GB. Aqui os tensores são views de um
mmap do arquivo (páginas de cache, descartáveis); load_state_dict copia para os parâmetros
e o pico anônimo cai para ~modelo + folga.
"""

from __future__ import annotations

import json
import mmap
import struct
import warnings

_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
}


def mmap_load_file(path: str, device: str = "cpu"):
    import torch

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    (n,) = struct.unpack("<Q", mm[:8])
    header = json.loads(mm[8:8 + n])
    base = 8 + n
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # frombuffer em buffer somente-leitura
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = getattr(torch, _DTYPES[meta["dtype"]])
            start, end = meta["data_offsets"]
            shape = tuple(meta["shape"])
            if end == start:
                out[name] = torch.empty(shape, dtype=dtype)
                continue
            t = torch.frombuffer(mm, dtype=dtype, count=(end - start) // torch.empty((), dtype=dtype).element_size(),
                                 offset=base + start)
            out[name] = t.reshape(shape)
    return out


def install() -> None:
    import laya.agent as agent

    agent.load_file = mmap_load_file


def build_model_lowmem(orig, dtype_name: str = "float16"):
    """Envolve build_model: sem init aleatório (load_state_dict strict sobrescreve tudo) e já em
    `dtype_name`. Evita o pico fp32 (421 M params = 1,7 GB); em bf16 o modelo ocupa ~0,85 GB."""
    def wrapped(*a, **kw):
        import torch
        import torch.nn.init as init

        names = [n for n in dir(init) if n.endswith("_") and not n.startswith("_")]
        saved = {n: getattr(init, n) for n in names}
        prev = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, dtype_name))
        for n in names:
            setattr(init, n, lambda t, *_a, **_k: t)
        try:
            model = orig(*a, **kw)
            lsd = model.load_state_dict
            # assign=True: os parâmetros passam a ser as views do mmap (sem cópia, páginas de cache)

            def load(sd, strict=True, **k):
                res = lsd(sd, strict=strict, assign=True, **k)
                # só o encoder (95% dos pesos) fica em meia precisão; heads/escalares voltam a fp32
                for child_name, child in model.named_children():
                    if child_name != "encoder":
                        child.float()
                model.temperature = model.temperature.float()
                return res

            model.load_state_dict = load
            return model
        finally:
            for n, f in saved.items():
                setattr(init, n, f)
            torch.set_default_dtype(prev)
    return wrapped
