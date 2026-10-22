"""lowmem.mmap_load_file: same tensors as safetensors.load_file, backed by mmap (no 2nd anon copy)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _file(tmp_path):
    data = {
        "a": torch.randn(4, 3),
        "b": torch.arange(10, dtype=torch.int64),
        "c": torch.randn(2, 2, dtype=torch.bfloat16),
        "d": torch.zeros(0),
    }
    path = tmp_path / "m.safetensors"
    st.save_file(data, str(path))
    return data, str(path)


def test_same_tensors_as_load_file(tmp_path):
    import lowmem

    data, path = _file(tmp_path)
    got = lowmem.mmap_load_file(path)
    assert set(got) == set(data)
    for k, v in data.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape
        assert torch.equal(got[k], v)


def test_load_state_dict_works_with_it(tmp_path):
    import lowmem

    lin = torch.nn.Linear(3, 4)
    path = tmp_path / "l.safetensors"
    st.save_file(lin.state_dict(), str(path))
    other = torch.nn.Linear(3, 4)
    other.load_state_dict(lowmem.mmap_load_file(str(path)), strict=True)
    assert torch.equal(other.weight, lin.weight)


def test_install_patches_laya_agent(monkeypatch):
    import lowmem

    agent = types.ModuleType("laya.agent")
    agent.load_file = None
    parent = types.ModuleType("laya")
    parent.agent = agent
    monkeypatch.setitem(sys.modules, "laya", parent)
    monkeypatch.setitem(sys.modules, "laya.agent", agent)
    lowmem.install()
    assert agent.load_file is lowmem.mmap_load_file


def test_build_wrapper_restores_global_state_and_assigns_fp16():
    import lowmem

    orig_init = torch.nn.init.normal_
    seen = {}

    def fake_build():
        seen["dtype"] = torch.get_default_dtype()
        seen["init_noop"] = torch.nn.init.normal_ is not orig_init
        return torch.nn.Linear(2, 2)

    m = lowmem.build_model_lowmem(fake_build)()
    assert seen == {"dtype": torch.float16, "init_noop": True}
    assert torch.get_default_dtype() == torch.float32
    assert torch.nn.init.normal_ is orig_init
    assert m.weight.dtype == torch.float16
