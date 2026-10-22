"""Auto-start tests: the MCP server brings its own container up.

These stop the Laya container, so they are opt-in: `pytest --run-slow`. A cold start with
the image already built took ~107s when this was written; a first-ever build that downloads
torch and the weights is far longer.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytestmark = pytest.mark.slow


def compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )


def container_is_running() -> bool:
    return bool(compose("ps", "-q", "laya-api").stdout.strip())


@pytest.fixture
def stopped_container():
    """Stop the stack for the test, and leave it running afterwards either way."""
    compose("down")
    assert not container_is_running(), "container still running after `docker compose down`"
    yield
    compose("up", "-d", "laya-api")


def test_predict_starts_the_container_when_it_is_down(laya, stopped_container):
    """With the stack down, a prediction boots it and still answers."""
    health = laya.call("laya_health")
    assert health["reachable"] is False, "API answered while the container was supposed to be down"
    assert health["auto_start"]["possible"] is True

    started = time.monotonic()
    result = laya.call(
        "laya_predict",
        state="Please refund the duplicate charge.",
        questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    )
    elapsed = time.monotonic() - started

    assert result["answers"]["refund"]["noul"] > 0.5
    assert container_is_running(), "prediction succeeded but no container is running"
    print(f"\ncold start through laya_predict: {elapsed:.1f}s")


def test_health_reports_the_api_as_down_without_starting_it(laya, stopped_container):
    """laya_health is a diagnostic: it must never launch anything by itself."""
    health = laya.call("laya_health")
    assert health["reachable"] is False
    assert not container_is_running(), "laya_health started the container"


def test_start_warms_the_stack_and_reports_the_cold_path(laya, stopped_container):
    """laya_start on a stopped stack reports a cold start and the real elapsed time."""
    result = laya.call("laya_start")
    assert result["ready"] is True
    assert result["already_running"] is False, "expected a cold start"
    assert result["elapsed_seconds"] > 1
    assert container_is_running()


def test_offline_hint_is_returned_when_auto_start_is_impossible(tmp_path):
    """With no compose file in reach, the failure must name the reason, not just fail."""
    from conftest import UV, SERVER, LayaClient
    from mcp import StdioServerParameters
    import os

    env = dict(os.environ)
    env["LAYA_COMPOSE_DIR"] = str(tmp_path)          # no docker-compose.yml here
    env["LAYA_API_URL"] = "http://localhost:59999"   # nothing listening

    client = LayaClient(StdioServerParameters(
        command=UV, args=["run", "--script", str(SERVER)], env=env,
    )).start()
    try:
        health = client.call("laya_health")
        assert health["reachable"] is False
        assert health["auto_start"]["possible"] is False
        assert "docker-compose.yml" in health["auto_start"]["blocked_because"]

        message = client.call_expecting_error(
            "laya_predict",
            state="anything",
            questions={"refund": {"type": "noul", "instructions": "refund?"}},
        )
        assert "Auto-start skipped" in message
        assert "docker-compose.yml" in message
    finally:
        client.stop()


def test_remote_api_url_never_triggers_a_local_container(tmp_path):
    """A remote LAYA_API_URL must not make the server start containers on this machine."""
    from conftest import UV, SERVER, LayaClient
    from mcp import StdioServerParameters
    import os

    env = dict(os.environ)
    env["LAYA_API_URL"] = "http://laya.internal.example:8000"

    client = LayaClient(StdioServerParameters(
        command=UV, args=["run", "--script", str(SERVER)], env=env,
    )).start()
    try:
        health = client.call("laya_health")
        assert health["reachable"] is False
        assert health["auto_start"]["possible"] is False
        assert "remote host" in health["auto_start"]["blocked_because"]
    finally:
        client.stop()
