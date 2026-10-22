"""Warm-up, health flag and latency header of app.py (stub laya, no container)."""

from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CALLS: list = []


class _StubRouter:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, questions, **kwargs):
        CALLS.append(kwargs)
        return {"answers": {"q": {"noul": 0.9}}}


@pytest.fixture()
def client(monkeypatch):
    CALLS.clear()
    monkeypatch.setenv("LAYA_WARMUP", "multilingual")
    stub = types.ModuleType("laya")
    stub.Router = _StubRouter
    monkeypatch.setitem(sys.modules, "laya", stub)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    with TestClient(app_module.app) as c:
        for _ in range(100):  # load + warm-up run in the background
            if c.get("/health").status_code == 200 and c.get("/health").json().get("warmed"):
                break
            time.sleep(0.05)
        yield c
    sys.modules.pop("app", None)


def test_warmup_runs_default_checkpoint_and_health_reports_it(client):
    assert client.get("/health").json()["warmed"] is True
    assert CALLS and CALLS[0].get("model") == "multilingual"


def test_latency_header_present(client):
    resp = client.get("/health")
    assert float(resp.headers["x-latency-ms"]) >= 0


def _load_app(monkeypatch, lowmem_env, agent_present=True):
    stub = types.ModuleType("laya")
    stub.Router = _StubRouter
    monkeypatch.setitem(sys.modules, "laya", stub)
    if agent_present:
        agent = types.ModuleType("laya.agent")
        agent.load_file = "orig"
        agent.build_model = lambda *a, **k: None
        stub.agent = agent
        monkeypatch.setitem(sys.modules, "laya.agent", agent)
    if lowmem_env is None:
        monkeypatch.delenv("LAYA_LOWMEM", raising=False)
    else:
        monkeypatch.setenv("LAYA_LOWMEM", lowmem_env)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("app", None)
    sys.modules.pop("lowmem", None)
    return importlib.import_module("app"), sys.modules.get("laya.agent")


def test_lowmem_on_by_default_patches_loader(monkeypatch):
    pytest.importorskip("torch")
    _, agent = _load_app(monkeypatch, None)
    assert agent.load_file != "orig"


def test_lowmem_can_be_disabled(monkeypatch):
    _, agent = _load_app(monkeypatch, "0")
    assert agent.load_file == "orig"


def test_lowmem_failure_does_not_break_startup(monkeypatch):
    module, _ = _load_app(monkeypatch, "1", agent_present=False)
    assert module.app is not None


def test_keepalive_touches_model_repeatedly(monkeypatch):
    import asyncio

    module, _ = _load_app(monkeypatch, "0")
    CALLS.clear()
    monkeypatch.setattr(module, "WARMUP_CHECKPOINT", "english")

    async def run():
        task = asyncio.create_task(module.keepalive(_StubRouter(), 0.01))
        await asyncio.sleep(0.15)
        task.cancel()

    asyncio.run(run())
    assert len(CALLS) >= 2 and CALLS[0] == {"model": "english"}
