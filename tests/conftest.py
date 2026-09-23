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
from pathlib import Path
from typing import Any, Callable

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "mcp" / "laya_mcp.py"
UV = os.environ.get("UV_BIN", str(Path.home() / ".local" / "bin" / "uv"))

CALL_TIMEOUT = float(os.environ.get("LAYA_TEST_TIMEOUT", "180"))


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests that stop and restart the Laya container (minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: restarts containers; needs --run-slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


class ToolCallError(RuntimeError):
    pass


class LayaClient:
    """Synchronous view over an MCP session running on a background event loop."""

    def __init__(self, params: StdioServerParameters):
        self._params = params
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

    def call(self, tool: str, **arguments: Any) -> Any:
        result = self._run(self._session.call_tool(tool, arguments))
        if result.is_error:
            raise ToolCallError(_text(result))
        return _payload(result)

    def call_expecting_error(self, tool: str, **arguments: Any) -> str:
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


@pytest.fixture(scope="session")
def laya():
    client = LayaClient(
        StdioServerParameters(
            command=UV,
            args=["run", "--script", str(SERVER)],
            env=dict(os.environ),
        )
    ).start()
    try:
        yield client
    finally:
        client.stop()


@pytest.fixture(scope="session")
def cases():
    with open(Path(__file__).parent / "fixtures" / "cases.json") as fh:
        return json.load(fh)
