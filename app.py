import asyncio
import logging
import os
import threading
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


# ---------- Carga enxuta dos pesos (LAYA_LOWMEM=0 desliga) ----------
lowmem_active = False


def _install_lowmem() -> bool:
    """Encoder fp16 via mmap, sem init aleatório: pico ~0,6 GB anônimo em vez de ~1,8 GB (ver lowmem.py).
    Retorna True só se o patch foi aplicado; falha vira WARNING (segue com a carga padrão)."""
    if os.environ.get("LAYA_LOWMEM", "1") == "0":
        return False
    try:
        import laya.agent as laya_agent
        import lowmem

        lowmem.install()
        laya_agent.build_model = lowmem.build_model_lowmem(laya_agent.build_model)
        logger.info("lowmem ativo (encoder fp16 mmap)")
        return True
    except Exception as exc:
        logger.warning(
            "lowmem NÃO aplicado (%s: %s): carga padrão, pico de RAM ~1,8 GB; /health lowmem_active=false",
            type(exc).__name__, exc, exc_info=True,
        )
        return False


lowmem_active = _install_lowmem()


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} inválido: {raw!r} (inteiro entre {lo} e {hi})") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} fora da faixa: {value} (use {lo}..{hi})")
    return value


def _env_bool01(name: str, default: int) -> bool:
    raw = os.environ.get(name, str(default)).strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{name} inválido: {raw!r} (use 0 ou 1)")
    return raw == "1"


# Quantos checkpoints ficam carregados (LRU). 1 = troca de idioma recarrega (~7,4 s CPU); 2 se couber no mem_limit.
LAYA_MAX_LOADED = _env_int("LAYA_MAX_LOADED", 1, 1, 4)
# 1 = carrega todos os checkpoints no startup (5-10x mais rápido em CPU, mais RAM). 0 = lazy.
LAYA_PRELOAD = _env_bool01("LAYA_PRELOAD", 0)
# Tamanho de contexto padrão do predict (english 512; multilingual até 8192). Vazio = default da laya.
LAYA_MAX_LEN_DEFAULT = _env_int("LAYA_MAX_LEN", 0, 0, 8192) or None  # 0 = não definido
MAX_LEN_RANGE = (1, 8192)


# ---------- Modelos de request/response ----------
class PredictRequest(BaseModel):
    state: Union[str, Dict[str, Any]] = Field(..., description="Texto, e-mail, ticket ou JSON")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Perguntas tipadas (choice, score, noul)")
    model: Optional[str] = Field(None, description="Forçar checkpoint: english | multilingual | typed-decisions")
    max_len: Optional[int] = Field(
        None, ge=MAX_LEN_RANGE[0], le=MAX_LEN_RANGE[1],
        description="Tamanho máximo de tokens do texto (1..8192; english usa até 512, multilingual até 8192)",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    warmed: bool = False
    lowmem_active: bool = False


# ---------- Ciclo de vida (carrega o modelo uma vez) ----------
router: Optional[Router] = None
router_error: Optional[str] = None
router_task: Optional[asyncio.Task] = None


# Checkpoint para texto de idioma indeterminado (curto, só palavras de conteúdo). "multilingual" é o default
# upstream da laya 0.4.x (decisão do usuário: última versão multilinguagem). Com ele a triagem PT fica em 0.500
# (ver PR #1). "english" dá mais acurácia em PT, mas só serve se o tráfego for majoritariamente inglês.
LAYA_DEFAULT_MODEL = os.environ.get("LAYA_DEFAULT_MODEL", "multilingual").strip().lower()
if LAYA_DEFAULT_MODEL not in ("english", "multilingual"):
    raise ValueError(f"LAYA_DEFAULT_MODEL inválido: {LAYA_DEFAULT_MODEL!r} (use english ou multilingual)")

WARMUP_CHECKPOINT = os.environ.get("LAYA_WARMUP", "")  # ex.: multilingual; vazio = desligado (carga do checkpoint estourou 4 GB no Docker)
_predict_lock = threading.Lock()  # modelo torch: uma inferência por vez, sem travar o event loop
warmed = False


async def warm_up(r):
    """Uma predição descartável carrega o checkpoint: sem isso a 1a chamada real leva ~20 s."""
    global warmed
    if not WARMUP_CHECKPOINT:
        return
    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(
            r.predict, "warm up", {"q": {"type": "noul", "instructions": "Is this text?"}},
            model=WARMUP_CHECKPOINT)
        warmed = True
        print(f"Warm-up {WARMUP_CHECKPOINT}: {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception:
        logger.exception("Falha no warm-up (segue sem)")


_keepalive_tasks: list = []  # referência forte (task solta pode ser coletada)
KEEPALIVE_S = float(os.environ.get("LAYA_KEEPALIVE", "60"))  # 0 = desligado


async def keepalive(r, interval=None):
    """Sob pressão de RAM do Docker VM o cache de páginas do mmap é despejado e a 1a chamada leva ~13 s:
    uma predição mínima periódica mantém os pesos quentes."""
    interval = KEEPALIVE_S if interval is None else interval
    while interval > 0:
        await asyncio.sleep(interval)
        def touch():
            with _predict_lock:  # lock dentro da thread: não trava o event loop
                r.predict("keepalive", {"q": {"type": "noul", "instructions": "Is this text?"}},
                          model=WARMUP_CHECKPOINT)

        try:
            await asyncio.to_thread(touch)
        except Exception:
            logger.exception("keepalive falhou")


async def load_router():
    global router, router_error
    print("Carregando Laya Router...", flush=True)
    try:
        loaded_router = await asyncio.to_thread(
            Router, preload=LAYA_PRELOAD, max_loaded=LAYA_MAX_LOADED, default=LAYA_DEFAULT_MODEL)
        router = loaded_router
        print("Router pronto.", flush=True)
        await warm_up(loaded_router)
        if WARMUP_CHECKPOINT and warmed:
            _keepalive_tasks.append(asyncio.create_task(keepalive(loaded_router)))
    except Exception as exc:
        router_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Falha ao carregar o Laya Router")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_task, router, router_error
    router = None
    router_error = None
    router_task = asyncio.create_task(load_router())
    yield
    if router_task and not router_task.done():
        router_task.cancel()
        await asyncio.gather(router_task, return_exceptions=True)
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
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.1f}"
    return Response(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=response.background,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    if router_error:
        raise HTTPException(status_code=503, detail=router_error)
    if router is None:
        raise HTTPException(status_code=503, detail="Router ainda inicializando")
    return {"status": "ok", "model_loaded": True, "warmed": warmed, "lowmem_active": lowmem_active}


@app.post("/predict")
def predict(req: PredictRequest):  # def (threadpool): não bloqueia /health durante a inferência
    if router is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado")

    try:
        kwargs = {}
        if req.model:
            kwargs["model"] = req.model
        max_len = req.max_len if req.max_len is not None else LAYA_MAX_LEN_DEFAULT
        if max_len is not None:
            kwargs["max_len"] = max_len

        with _predict_lock:
            result = router.predict(req.state, req.questions, **kwargs)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
