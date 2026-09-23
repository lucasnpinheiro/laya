# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp>=2.0.0",
#     "httpx>=0.27",
# ]
# ///
"""
Laya MCP server.

Exposes the Laya System 1 decision model (https://huggingface.co/convaiinnovations/laya)
to MCP clients. Talks HTTP to the Laya Decision API (the FastAPI container in this repo),
so the model stays resident in one process instead of being rebuilt per call.

If the API is not answering and it is supposed to run locally, the server brings the
container up itself with `docker compose up -d` and waits for the model to load.

Environment:
    LAYA_API_URL       base URL of the Laya Decision API (default: http://localhost:8000)
    LAYA_TIMEOUT       prediction request timeout in seconds (default: 120)
    LAYA_AUTO_START    1/0 - start the compose stack when the API is down (default: 1)
    LAYA_COMPOSE_DIR   directory holding docker-compose.yml (default: this file's parent repo)
    LAYA_COMPOSE_SERVICE  service to start (default: laya-api)
    LAYA_START_TIMEOUT seconds to wait for the model to load after boot (default: 900)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

API_URL = os.environ.get("LAYA_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("LAYA_TIMEOUT", "120"))

AUTO_START = os.environ.get("LAYA_AUTO_START", "1").lower() not in ("0", "false", "no")
COMPOSE_DIR = Path(os.environ.get("LAYA_COMPOSE_DIR") or Path(__file__).resolve().parent.parent)
COMPOSE_SERVICE = os.environ.get("LAYA_COMPOSE_SERVICE", "laya-api")
START_TIMEOUT = float(os.environ.get("LAYA_START_TIMEOUT", "900"))

CHECKPOINTS = ("english", "multilingual", "typed-decisions")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}

OFFLINE_HINT = (
    f"Laya Decision API unreachable at {API_URL}. "
    f"Start it with `docker compose up -d` in {COMPOSE_DIR}, then check "
    "`curl -s $LAYA_API_URL/health`. The first boot downloads the model weights and can "
    "take several minutes."
)

# Serialises boot attempts so concurrent tool calls do not each launch compose.
_boot_lock = asyncio.Lock()
# Set once the API has answered healthy, so the happy path skips the probe entirely.
_api_ready = False


def _is_local_api() -> bool:
    return (urlparse(API_URL).hostname or "") in _LOCAL_HOSTS


def _can_auto_start() -> tuple[bool, str]:
    """Whether auto-start is possible here, and why not when it is not."""
    if not AUTO_START:
        return False, "auto-start disabled via LAYA_AUTO_START=0"
    if not _is_local_api():
        return False, f"LAYA_API_URL points at a remote host ({API_URL}); not starting containers for it"
    if shutil.which("docker") is None:
        return False, "the `docker` binary is not on PATH"
    if not (COMPOSE_DIR / "docker-compose.yml").exists() and not (COMPOSE_DIR / "compose.yml").exists():
        return False, f"no docker-compose.yml found in {COMPOSE_DIR} (set LAYA_COMPOSE_DIR)"
    return True, ""


async def _probe_health(timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Return the /health payload, or None when the API is not answering."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{API_URL}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


async def _compose_up() -> tuple[bool, str]:
    """Run `docker compose up -d <service>`. Returns (ok, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "compose", "up", "-d", COMPOSE_SERVICE,
        cwd=str(COMPOSE_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=START_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return False, f"`docker compose up -d {COMPOSE_SERVICE}` exceeded {START_TIMEOUT:.0f}s (image build?)"

    return proc.returncode == 0, (out or b"").decode(errors="replace").strip()


async def _wait_until_loaded(deadline: float) -> Optional[Dict[str, Any]]:
    """Poll /health until the model reports loaded, or the deadline passes."""
    while time.monotonic() < deadline:
        health = await _probe_health()
        if health and health.get("model_loaded"):
            return health
        await asyncio.sleep(3)
    return None


async def _ensure_api() -> None:
    """Make sure the API is up and the model is loaded, starting the stack if needed.

    Cheap after the first success: a module-level flag short-circuits the probe.
    """
    global _api_ready
    if _api_ready:
        return

    async with _boot_lock:
        if _api_ready:
            return

        health = await _probe_health()
        if health and health.get("model_loaded"):
            _api_ready = True
            return

        deadline = time.monotonic() + START_TIMEOUT

        if health is not None:
            # API answering but still building the Router - just wait it out.
            if await _wait_until_loaded(deadline):
                _api_ready = True
                return
            raise ToolError(
                f"Laya API at {API_URL} is up but the model did not finish loading within "
                f"{START_TIMEOUT:.0f}s. Check `docker compose logs -f {COMPOSE_SERVICE}`."
            )

        ok, why_not = _can_auto_start()
        if not ok:
            raise ToolError(f"{OFFLINE_HINT}\n\nAuto-start skipped: {why_not}.")

        started, output = await _compose_up()
        if not started:
            raise ToolError(
                f"Failed to start the Laya stack with `docker compose up -d {COMPOSE_SERVICE}` "
                f"in {COMPOSE_DIR}.\n\n{output[-2000:]}"
            )

        if await _wait_until_loaded(deadline):
            _api_ready = True
            return

        raise ToolError(
            f"Started the Laya container but {API_URL}/health never reported the model loaded "
            f"within {START_TIMEOUT:.0f}s. The first boot downloads ~800MB of weights; watch "
            f"`docker compose logs -f {COMPOSE_SERVICE}` and retry."
        )

mcp = MCPServer(
    "laya",
    instructions=(
        "Laya is a non-autoregressive System 1 decision model: it answers typed questions "
        "about a state (text, email, ticket, JSON) in a single forward pass (~33ms) with "
        "calibrated probabilities, across 100+ languages. It never generates free text.\n\n"
        "Use laya_predict for classification, triage, routing, scoring, guardrails and "
        "boolean judgements over a piece of content — it is far cheaper and faster than "
        "reasoning about it yourself, and it returns confidences you can threshold.\n\n"
        "Question types: 'choice' (pick one option), 'score' (ordinal level), 'noul' "
        "(probability of yes). Keep choice questions under ~20 options; above that, split "
        "into a coarse-to-fine hierarchy.\n\n"
        "The backing container is started automatically on the first call. A cold start "
        "downloads the weights and can take minutes; every call after that is milliseconds."
    ),
)


async def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    global _api_ready

    await _ensure_api()

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{API_URL}{path}", json=payload)
    except httpx.RequestError:
        # The container died or was restarted since the last successful call.
        # Drop the cached readiness, let _ensure_api bring it back, and retry once.
        _api_ready = False
        await _ensure_api()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(f"{API_URL}{path}", json=payload)
        except httpx.RequestError as exc:
            _api_ready = False
            raise ToolError(f"{OFFLINE_HINT}\n\nUnderlying error: {exc!r}") from exc

    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise ToolError(f"Laya API returned HTTP {resp.status_code}: {detail}")

    return resp.json()


def _validate_questions(questions: Dict[str, Dict[str, Any]]) -> None:
    if not isinstance(questions, dict) or not questions:
        raise ToolError("`questions` must be a non-empty object keyed by question name.")

    for name, q in questions.items():
        if not isinstance(q, dict):
            raise ToolError(f"Question '{name}' must be an object.")

        qtype = q.get("type")
        if qtype not in ("choice", "score", "noul"):
            raise ToolError(
                f"Question '{name}' has type {qtype!r}; must be 'choice', 'score' or 'noul'."
            )
        if not q.get("instructions"):
            raise ToolError(f"Question '{name}' is missing `instructions`.")

        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ToolError(
                    f"Question '{name}' is type 'choice' and needs `criteria` as an object "
                    "of at least two {option: description} pairs."
                )
        elif qtype == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ToolError(
                    f"Question '{name}' is type 'score' and needs `criteria` as an ordered "
                    "list of at least two level descriptions, lowest first."
                )
        elif criteria is not None:
            raise ToolError(f"Question '{name}' is type 'noul' and must not carry `criteria`.")


@mcp.tool()
async def laya_predict(
    state: Union[str, Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
    model: Optional[Literal["english", "multilingual", "typed-decisions"]] = None,
) -> Dict[str, Any]:
    """Answer typed questions about a state with calibrated probabilities, in one forward pass.

    Args:
        state: The content to judge. A plain string, or an object of named fields
            (e.g. {"from": ..., "subject": ..., "body": ...}) for richer context.
        questions: Object keyed by question name. Each value is:
            - {"type": "choice", "instructions": str, "criteria": {option: description, ...}}
              -> returns the chosen option plus a confidence.
            - {"type": "score", "instructions": str, "criteria": [lowest, ..., highest]}
              -> returns a float on the ordinal scale.
            - {"type": "noul", "instructions": str}
              -> returns a probability in [0, 1] that the answer is yes.
            All questions are answered together in a single pass, so ask everything at once.
        model: Force a checkpoint. Omit to let the router pick by script/language detection:
            'english' (ModernBERT-large, English text, guardrails, email triage),
            'multilingual' (mmBERT-base, 100+ languages, ~2.2x faster),
            'typed-decisions' (fine-tuned on invoice/security/support/agent-trace workflows).

    Returns:
        {"answers": {name: {...}}, "routing": {"model", "repo", "reason"}}.

    Notes:
        The model ships over-confident; treat probabilities as ranking signals unless you
        have fitted a temperature on your own data. Non-Latin scripts must not use the
        'english' checkpoint — it stays confident while being wrong.
    """
    _validate_questions(questions)

    payload: Dict[str, Any] = {"state": state, "questions": questions}
    if model:
        payload["model"] = model

    return await _post("/predict", payload)


@mcp.tool()
async def laya_classify(
    text: str,
    labels: Dict[str, str],
    instructions: str = "Which label best describes this content?",
    model: Optional[Literal["english", "multilingual", "typed-decisions"]] = None,
) -> Dict[str, Any]:
    """Single-label classification shortcut over laya_predict.

    Use for a plain "which bucket does this fall into" question. For anything richer —
    several questions at once, scores, boolean judgements — use laya_predict instead,
    since it answers them all in the same forward pass.

    Args:
        text: Content to classify.
        labels: {label: short description of what belongs in it}. Keep under ~20 labels;
            option text shares a fixed token budget, so many labels blur together.
        instructions: The question put to the model.
        model: Force a checkpoint; omit to let the router decide.

    Returns:
        {"label": str, "confidence": float, "raw": <full laya_predict result>}.
    """
    if len(labels) < 2:
        raise ToolError("`labels` needs at least two entries.")

    result = await laya_predict(
        state=text,
        questions={"label": {"type": "choice", "instructions": instructions, "criteria": labels}},
        model=model,
    )

    answer = result.get("answers", {}).get("label", {})
    return {
        "label": answer.get("choice"),
        "confidence": answer.get("confidence"),
        "raw": result,
    }


@mcp.tool()
async def laya_health() -> Dict[str, Any]:
    """Report whether the Laya Decision API is up and the model is loaded. Starts nothing.

    The prediction tools bring the container up on their own, so this is a diagnostic:
    use it to see why a call is slow or failing, or to confirm the stack before a batch.
    A cold API answers while the weights are still downloading, with model_loaded=false.
    """
    can_start, why_not = _can_auto_start()
    info: Dict[str, Any] = {
        "api_url": API_URL,
        "auto_start": {
            "enabled": AUTO_START,
            "possible": can_start,
            "compose_dir": str(COMPOSE_DIR),
            "service": COMPOSE_SERVICE,
        },
    }
    if not can_start:
        info["auto_start"]["blocked_because"] = why_not

    health = await _probe_health(timeout=10)
    if health is None:
        return {"reachable": False, "hint": OFFLINE_HINT, **info}

    return {"reachable": True, **health, **info}


@mcp.tool()
async def laya_start() -> Dict[str, Any]:
    """Bring the Laya container up and wait until the model is loaded.

    Only needed to warm the stack deliberately — laya_predict and laya_classify already
    do this on demand. A cold first boot pulls ~800MB of weights and can take minutes;
    afterwards the cached volume makes it quick.
    """
    global _api_ready

    started_at = time.monotonic()
    already = await _probe_health()
    was_up = bool(already and already.get("model_loaded"))

    if not was_up:
        # The cached readiness flag can outlive the container (a `docker compose down`
        # between calls). Drop it so _ensure_api actually boots instead of trusting it.
        _api_ready = False

    await _ensure_api()

    return {
        "ready": True,
        "already_running": was_up,
        "api_url": API_URL,
        "compose_dir": str(COMPOSE_DIR),
        "service": COMPOSE_SERVICE,
        "elapsed_seconds": round(time.monotonic() - started_at, 1),
    }


@mcp.tool()
def laya_checkpoints() -> List[Dict[str, Any]]:
    """List the Laya checkpoints, their backbones, context budgets and what each is best at.

    Read this before forcing a `model` on laya_predict. Leaving `model` unset is usually
    right: the router detects script and language in under a millisecond and dispatches.
    """
    return [
        {
            "name": "english",
            "backbone": "ModernBERT-large",
            "params": "421M",
            "context": 512,
            "head_max_len": 192,
            "best_at": "English text, guardrails, email triage",
            "warning": "Collapses on non-Latin scripts while staying confident. Never force it on those.",
        },
        {
            "name": "multilingual",
            "backbone": "mmBERT-base",
            "params": "322M",
            "context": 1024,
            "head_max_len": 256,
            "best_at": "100+ languages, ~2.2x faster than the English checkpoint",
            "warning": "Encoder supports up to 8192 tokens, but the served default is 1024.",
        },
        {
            "name": "typed-decisions",
            "backbone": "ModernBERT-large",
            "params": "421M",
            "context": 1024,
            "head_max_len": 256,
            "best_at": "invoice processing, security incidents, customer service, agent-trace observability",
            "warning": "Fine-tuned on those four workflows; no advantage outside them.",
        },
    ]


if __name__ == "__main__":
    mcp.run()
