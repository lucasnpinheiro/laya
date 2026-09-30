import logging
import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Union
from contextlib import asynccontextmanager
from laya import Router


# ---------- Log de request/response (stdout -> docker logs) ----------
LOG_BODY_MAX = int(os.environ.get("LOG_BODY_MAX", "10000"))

logger = logging.getLogger("laya.http")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


def _fmt_body(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if len(text) > LOG_BODY_MAX:
        return f"{text[:LOG_BODY_MAX]}... [truncated {len(text) - LOG_BODY_MAX} chars]"
    return text


# ---------- Modelos de request/response ----------
class PredictRequest(BaseModel):
    state: Union[str, Dict[str, Any]] = Field(..., description="Texto, e-mail, ticket ou JSON")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Perguntas tipadas (choice, score, noul)")
    model: Optional[str] = Field(None, description="Forçar checkpoint: english | multilingual | typed-decisions")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------- Ciclo de vida (carrega o modelo uma vez) ----------
router: Optional[Router] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    print("Carregando Laya Router...")
    router = Router(preload=True, max_loaded=1)
    print("Router pronto.")
    yield
    # cleanup se necessário
    router = None


app = FastAPI(
    title="Laya Decision API",
    description="API HTTP para o modelo Laya (System 1 Decision Model)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_request_response(request: Request, call_next):
    started = time.perf_counter()
    req_body = await request.body()
    client = f"{request.client.host}:{request.client.port}" if request.client else "-"
    path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    logger.info("--> %s %s client=%s body=%s", request.method, path, client, _fmt_body(req_body))

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("<-- %s %s status=500 unhandled error", request.method, path)
        raise

    resp_body = b"".join([chunk async for chunk in response.body_iterator])
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "<-- %s %s status=%d time=%.1fms body=%s",
        request.method, path, response.status_code, elapsed_ms, _fmt_body(resp_body),
    )
    return Response(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=response.background,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "model_loaded": router is not None}


@app.post("/predict")
async def predict(req: PredictRequest):
    if router is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado")

    try:
        kwargs = {}
        if req.model:
            kwargs["model"] = req.model

        result = router.predict(req.state, req.questions, **kwargs)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
