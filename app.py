from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Union
from contextlib import asynccontextmanager
from laya import Router


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
