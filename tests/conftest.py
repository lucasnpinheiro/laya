"""Shared MCP client plumbing for the Laya test battery.

The MCP client is async, but its anyio context managers must be entered and exited in the
same task. pytest-asyncio finalises async fixtures in a different task, which anyio rejects
("Attempted to exit cancel scope in a different task"). So the client lives in a dedicated
thread with its own event loop, opened and closed there, and the tests stay plain and
synchronous. One server process serves the whole run; the model itself is in the container.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pytest
from prediction_cache import PredictionCache, make_key
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "mcp" / "laya_mcp.py"
UV = os.environ.get("UV_BIN", str(Path.home() / ".local" / "bin" / "uv"))

CALL_TIMEOUT = float(os.environ.get("LAYA_TEST_TIMEOUT", "180"))


API_URL = os.environ.get("LAYA_API_URL", "http://localhost:8000").rstrip("/")
WARM_WAIT = float(os.environ.get("LAYA_TEST_WARM_WAIT", "90"))
REPORT_DIR = Path(__file__).parent / "reports"

# Tools whose answers are deterministic model output; everything else always hits the server.
_MEMOIZED_TOOLS = {"laya_predict", "laya_classify"}


def pytest_addoption(parser):
    parser.addoption("--model", action="store_true", default=False,
                     help="also run the model tier (real predictions over the measured banks)")
    parser.addoption("--bench", action="store_true", default=False,
                     help="also run the bench tier (full measurement of the question banks, ~10 min)")
    parser.addoption("--all", action="store_true", default=False, dest="all_tiers",
                     help="fast + model + bench tiers (everything except the container-restarting slow tier)")
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="also run tests that stop and restart the Laya container (minutes)")
    parser.addoption("--write-reports", action="store_true", default=False,
                     help="rewrite tests/reports/*_metrics.json (same as LAYA_WRITE_REPORTS=1)")


def pytest_configure(config):
    config.addinivalue_line("markers", "fast: no model call (contract, stubs, static files); default tier")
    config.addinivalue_line("markers", "model: real predictions; selected with --model or --all")
    config.addinivalue_line("markers", "bench: full bank measurements that write the metric reports; needs --bench or --all")
    config.addinivalue_line("markers", "slow: restarts containers; needs --run-slow")


def pytest_collection_modifyitems(config, items):
    want_model = config.getoption("--model") or config.getoption("all_tiers")
    want_bench = config.getoption("--bench") or config.getoption("all_tiers")
    want_slow = config.getoption("--run-slow")
    kept, dropped = [], []
    for item in items:
        if "slow" in item.keywords and not want_slow:
            dropped.append(item)
        elif "bench" in item.keywords and not want_bench:
            dropped.append(item)
        elif "model" in item.keywords and not (want_model or want_slow):
            dropped.append(item)
        else:
            if not {"model", "bench", "slow"} & set(item.keywords):
                item.add_marker(pytest.mark.fast)
                if item.get_closest_marker("timeout") is None:
                    item.add_marker(pytest.mark.timeout(60))
            kept.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def reports_enabled(config=None) -> bool:
    if os.environ.get("LAYA_WRITE_REPORTS", "").lower() in ("1", "true", "yes"):
        return True
    return bool(config and config.getoption("--write-reports"))


@pytest.fixture(scope="session")
def write_report(request):
    """Return write(name, metrics): writes tests/reports/<name> only when reports are enabled."""
    enabled = reports_enabled(request.config)

    def write(name: str, metrics) -> None:
        if not enabled:
            return
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / name).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    return write


def _health():
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def wait_until_warmed() -> None:
    """Fail fast with a clear message instead of hanging on a cold or missing server."""
    deadline = time.monotonic() + WARM_WAIT
    health = _health()
    while not (health.get("model_loaded") and health.get("warmed")):
        if time.monotonic() >= deadline:
            pytest.exit(
                f"Laya API at {API_URL} is not warmed after {WARM_WAIT:.0f}s (last /health: {health}). "
                "Start it with `docker compose up -d`, wait for warmed=true, or raise LAYA_TEST_WARM_WAIT.",
                returncode=3,
            )
        time.sleep(2)
        health = _health()


class ToolCallError(RuntimeError):
    pass


class LayaClient:
    """Synchronous view over an MCP session running on a background event loop."""

    def __init__(self, params: StdioServerParameters, cache: PredictionCache | None = None):
        self._params = params
        self.cache = cache or PredictionCache()
        self.salt = ""
        self.real_calls = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="laya-mcp")

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> "LayaClient":
        self._thread.start()
        self._ready.wait(timeout=120)
        if self._startup_error is not None:
            raise RuntimeError("MCP server failed to start") from self._startup_error
        if self._session is None:
            raise RuntimeError("MCP server did not become ready within 120s")
        return self

    def stop(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=30)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            self._startup_error = exc
            self._ready.set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=CALL_TIMEOUT)

    # -- API --------------------------------------------------------------------------

    def list_tools(self):
        return self._run(self._session.list_tools()).tools

    def call_fresh(self, tool: str, **arguments: Any) -> Any:
        """Always hits the server (determinism, latency and contract checks)."""
        self.real_calls += 1
        result = self._run(self._session.call_tool(tool, arguments))
        if result.is_error and "unreachable" in _text(result):
            # Container restarted under us: wait for warm-up once, then retry (fails loudly if not).
            wait_until_warmed()
            result = self._run(self._session.call_tool(tool, arguments))
        if result.is_error:
            raise ToolCallError(_text(result))
        return _payload(result)

    def call(self, tool: str, **arguments: Any) -> Any:
        """Identical deterministic predictions are answered from the session memo."""
        if tool not in _MEMOIZED_TOOLS:
            return self.call_fresh(tool, **arguments)
        key = make_key(tool, arguments, salt=self.salt)
        return self.cache.get_or_compute(key, lambda: self.call_fresh(tool, **arguments))

    def call_expecting_error(self, tool: str, **arguments: Any) -> str:
        self.real_calls += 1
        result = self._run(self._session.call_tool(tool, arguments))
        assert result.is_error, f"expected {tool} to fail, got: {_text(result)[:300]}"
        return _text(result)


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


def _payload(result) -> Any:
    """Prefer structured output; fall back to decoding the text blocks.

    A tool returning a list arrives as one content block per item, so blocks are decoded
    individually and collapsed back to a single value when there is only one.
    """
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        # MCPServer wraps non-object returns under "result".
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured

    decoded = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            decoded.append(json.loads(text))
        except json.JSONDecodeError:
            decoded.append(text)
    return decoded[0] if len(decoded) == 1 else decoded


_STATS: dict = {}


def pytest_terminal_summary(terminalreporter):
    if _STATS:
        terminalreporter.write_line(
            f"[laya] server calls (incl. contract): {_STATS['real']}, memoized hits: {_STATS['hits']}")


def _build_cache() -> PredictionCache:
    if os.environ.get("LAYA_TEST_DISK_CACHE", "").lower() in ("1", "true", "yes"):
        return PredictionCache(disk_dir=REPO_ROOT / ".test-cache")
    return PredictionCache()


@pytest.fixture(scope="session")
def laya():
    wait_until_warmed()
    client = LayaClient(
        StdioServerParameters(
            command=UV,
            args=["run", "--script", str(SERVER)],
            # Never let a test spawn its own compose stack (it competes with the real container).
            env={**os.environ, "LAYA_AUTO_START": "0"},
        ),
        cache=_build_cache(),
    ).start()
    # Disk entries are only valid for the same model configuration.
    client.salt = json.dumps([os.environ.get("LAYA_LOWMEM", ""), _health(),
                              client.call_fresh("laya_checkpoints")], sort_keys=True, default=str)
    try:
        yield client
    finally:
        _STATS.update(real=client.real_calls, hits=client.cache.hits)
        client.stop()


@pytest.fixture(scope="session")
def cases():
    with open(Path(__file__).parent / "fixtures" / "cases.json") as fh:
        return json.load(fh)
