"""Env knobs of app.py: LAYA_MAX_LOADED, LAYA_PRELOAD, LAYA_MAX_LEN, lowmem_active in /health.

Stub laya module, no container or weights:
    uv run --with fastapi --with httpx --with pytest --no-project pytest tests/test_tuning.py
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTER_KWARGS: list = []
PREDICT_KWARGS: list = []


class _StubRouter:
    def __init__(self, *args, **kwargs):
        ROUTER_KWARGS.append(kwargs)

    def predict(self, state, questions, **kwargs):
        PREDICT_KWARGS.append(kwargs)
        return {"answers": {"q": {"label": "yes", "confidence": 0.9}}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("LAYA_MAX_LOADED", "LAYA_PRELOAD", "LAYA_MAX_LEN", "LAYA_LOWMEM", "LAYA_WARMUP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYA_LOWMEM", "0")  # no torch/laya.agent needed here
    ROUTER_KWARGS.clear()
    PREDICT_KWARGS.clear()


def _import_app(monkeypatch):
    stub = types.ModuleType("laya")
    stub.Router = _StubRouter
    monkeypatch.setitem(sys.modules, "laya", stub)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("app", None)
    return importlib.import_module("app")


@pytest.fixture()
def client_factory(monkeypatch):
    clients = []

    def make(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        app_module = _import_app(monkeypatch)
        c = TestClient(app_module.app)
        c.__enter__()
        clients.append(c)
        for _ in range(200):
            if c.get("/health").status_code == 200:
                break
            time.sleep(0.02)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)
    sys.modules.pop("app", None)


PAYLOAD = {"state": "cobrado duas vezes", "questions": {"q": {"type": "noul", "text": "reclamação?"}}}


# ---- LAYA_MAX_LOADED / LAYA_PRELOAD -------------------------------------------------

def test_router_defaults_are_lazy_and_single_checkpoint(client_factory):
    client_factory()
    assert ROUTER_KWARGS[0]["max_loaded"] == 1
    assert ROUTER_KWARGS[0]["preload"] is False


def test_router_reads_max_loaded_and_preload_from_env(client_factory):
    client_factory(LAYA_MAX_LOADED="2", LAYA_PRELOAD="1")
    assert ROUTER_KWARGS[0]["max_loaded"] == 2
    assert ROUTER_KWARGS[0]["preload"] is True


@pytest.mark.parametrize("name,value", [
    ("LAYA_MAX_LOADED", "0"), ("LAYA_MAX_LOADED", "abc"), ("LAYA_MAX_LOADED", "-1"),
    ("LAYA_PRELOAD", "yes"), ("LAYA_PRELOAD", "2"),
])
def test_invalid_tuning_env_fails_at_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _import_app(monkeypatch)
    sys.modules.pop("app", None)


# ---- max_len ------------------------------------------------------------------------

def test_predict_passes_max_len_to_router(client_factory):
    c = client_factory()
    resp = c.post("/predict", json={**PAYLOAD, "max_len": 2048})
    assert resp.status_code == 200
    assert PREDICT_KWARGS[-1]["max_len"] == 2048


def test_predict_without_max_len_does_not_send_it(client_factory):
    c = client_factory()
    assert c.post("/predict", json=PAYLOAD).status_code == 200
    assert "max_len" not in PREDICT_KWARGS[-1]


@pytest.mark.parametrize("value", [0, -5, 8193, "abc"])
def test_predict_rejects_out_of_range_max_len(client_factory, value):
    c = client_factory()
    resp = c.post("/predict", json={**PAYLOAD, "max_len": value})
    assert resp.status_code == 422
    assert PREDICT_KWARGS == []


def test_env_max_len_is_default_and_request_overrides(client_factory):
    c = client_factory(LAYA_MAX_LEN="512")
    c.post("/predict", json=PAYLOAD)
    assert PREDICT_KWARGS[-1]["max_len"] == 512
    c.post("/predict", json={**PAYLOAD, "max_len": 1024})
    assert PREDICT_KWARGS[-1]["max_len"] == 1024


def test_invalid_env_max_len_fails_at_startup(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_LEN", "9000")
    with pytest.raises(ValueError, match="LAYA_MAX_LEN"):
        _import_app(monkeypatch)
    sys.modules.pop("app", None)


# ---- lowmem_active / warning --------------------------------------------------------

def test_health_reports_lowmem_inactive_when_disabled(client_factory):
    c = client_factory()
    assert c.get("/health").json()["lowmem_active"] is False


def test_lowmem_failure_logs_warning_and_reports_inactive(client_factory, monkeypatch, caplog):
    # LAYA_LOWMEM=1 with a stub laya that has no `agent` submodule: patch fails.
    caplog.set_level(logging.WARNING, logger="laya.http")
    monkeypatch.setenv("LAYA_LOWMEM", "1")
    c = client_factory(LAYA_LOWMEM="1")
    health = c.get("/health").json()
    assert health["lowmem_active"] is False
    assert health["status"] == "ok"  # healthcheck still passes
    assert any(r.levelno == logging.WARNING and "lowmem" in r.getMessage().lower() for r in caplog.records)
