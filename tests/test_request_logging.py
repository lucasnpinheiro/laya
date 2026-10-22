"""Request/response logging middleware of app.py.

Runs app.py in-process with a stub `laya` module, so no container or weights are needed:
    uv run --with fastapi --with httpx --with pytest --no-project pytest tests/test_request_logging.py
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


class _StubRouter:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, questions, **kwargs):
        return {"answers": {"q": {"label": "yes", "confidence": 0.9}}}


@pytest.fixture()
def client(monkeypatch):
    stub = types.ModuleType("laya")
    stub.Router = _StubRouter
    monkeypatch.setitem(sys.modules, "laya", stub)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    with TestClient(app_module.app) as c:
        yield c
    sys.modules.pop("app", None)


def test_logs_request_and_response_bodies(client, caplog):
    caplog.set_level(logging.INFO, logger="laya.http")
    payload = {"state": "cobrado duas vezes", "questions": {"q": {"type": "noul", "text": "reclamação?"}}}

    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    assert resp.json()["answers"]["q"]["label"] == "yes"  # body still reaches the client
    text = caplog.text
    assert "POST /predict" in text
    assert "cobrado duas vezes" in text
    assert "status=200" in text
    assert '"label":"yes"' in text or '"label": "yes"' in text


def test_logs_error_responses(client, caplog):
    caplog.set_level(logging.INFO, logger="laya.http")

    resp = client.post("/predict", json={"state": "x"})

    assert resp.status_code == 422
    assert "status=422" in caplog.text


def test_truncates_large_bodies(client, caplog, monkeypatch):
    import app

    monkeypatch.setattr(app, "LOG_BODY_MAX", 50)
    caplog.set_level(logging.INFO, logger="laya.http")

    client.post("/predict", json={"state": "a" * 500, "questions": {}})

    assert "a" * 500 not in caplog.text
    assert "truncated" in caplog.text
