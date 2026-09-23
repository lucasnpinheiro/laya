# Laya — System 1 Decision Model (Docker)

Rodar o modelo [Laya](https://huggingface.co/convaiinnovations/laya) em container, via CPU,
exposto como API HTTP (FastAPI) ou como script único.

O repositório está configurado no **modo API**. O modo script está documentado abaixo caso
você prefira rodar um batch isolado.

## Estrutura

| Arquivo | Função |
|---|---|
| `Dockerfile` | Python 3.11 slim + PyTorch CPU + `laya` + FastAPI/uvicorn, usuário não-root `laya` (uid 10001) |
| `app.py` | API FastAPI com `/health` e `/predict` |
| `docker-compose.yml` | Serviço `laya-api` na porta 8000 + volume de cache do Hugging Face |
| `mcp/laya_mcp.py` | Servidor MCP para Claude Code, com auto-start do container |
| `tests/` | Bateria de testes + documentação dos resultados medidos |
| `question_banks/spec_research.json` | Perguntas validadas para triagem de requisitos, com vereditos de aprovação/reprovação |

O volume `laya-hf-cache` mantém os pesos do modelo em `/home/laya/.cache/huggingface`,
evitando novo download a cada restart.

---

## Modo API (padrão)

### Subir

```bash
docker compose up --build -d
```

O primeiro boot baixa o modelo — pode demorar. Por isso o healthcheck tem
`start_period: 60s`.

### Acompanhar

```bash
# Ver logs
docker compose logs -f

# Status
curl http://localhost:8000/health
```

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "model_loaded": true}` |
| `POST` | `/predict` | Executa a predição tipada |
| `GET` | `/docs` | Swagger UI interativo |

### Corpo do `/predict`

```jsonc
{
  "state": "texto | objeto JSON (e-mail, ticket, etc.)",
  "questions": { "<nome>": { "type": "choice|score|noul", "instructions": "...", "criteria": ... } },
  "model": "english | multilingual | typed-decisions"   // opcional
}
```

Tipos de pergunta:

- `choice` — `criteria` é um objeto `{ opção: descrição }`.
- `score` — `criteria` é uma lista ordenada de níveis.
- `noul` — booleano; não usa `criteria`.

### Health check

```bash
curl -s http://localhost:8000/health | jq
```

### Predição completa (inglês)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "from": "user@acme.com",
      "subject": "Duplicate charge on invoice #4411",
      "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs, outages, system errors",
          "sales": "pricing, new contracts",
          "other": "everything else"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
      },
      "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
      },
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
      }
    }
  }' | jq
```

### Predição em português (checkpoint `multilingual`)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "body": "Fui cobrado duas vezes na fatura de março. Quero o reembolso hoje ou vou cancelar o plano."
    },
    "questions": {
      "departamento": {
        "type": "choice",
        "instructions": "Qual departamento deve tratar este pedido?",
        "criteria": {
          "billing": "faturas, pagamentos, reembolsos",
          "technical": "bugs e problemas técnicos",
          "sales": "preços e novos contratos",
          "other": "qualquer outra coisa"
        }
      },
      "urgencia": {
        "type": "score",
        "instructions": "Qual o nível de urgência?",
        "criteria": ["baixa", "média", "alta", "crítica"]
      },
      "risco_churn": {
        "type": "noul",
        "instructions": "O cliente ameaça cancelar?"
      },
      "pede_reembolso": {
        "type": "noul",
        "instructions": "O cliente pede reembolso explicitamente?"
      }
    },
    "model": "multilingual"
  }' | jq
```

### Predição mínima (`state` como string)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Por favor reembolse a cobrança duplicada.",
    "questions": {
      "refund": {
        "type": "noul",
        "instructions": "O cliente está pedindo reembolso?"
      }
    }
  }' | jq
```

### Dicas rápidas

- Documentação interativa: abra <http://localhost:8000/docs> (Swagger UI).
- Para forçar um checkpoint específico use `"model": "english" | "multilingual" | "typed-decisions"`.
- O volume `laya-hf-cache` evita baixar o modelo toda vez que reiniciar o container.
- Carregar um subfolder direto pela lib:

  ```python
  from laya import load
  agent = load("convaiinnovations/laya", subfolder="multilingual")
  ```

---

## MCP para Claude Code

`mcp/laya_mcp.py` expõe o Laya como servidor MCP (stdio). Ele conversa por HTTP com a API
acima e **sobe o container sozinho** na primeira chamada, caso não esteja rodando.

### Registrar

```bash
claude mcp add --scope user laya \
  "$(which uv)" -- run --script /Users/lucasnpinheiro/PhpstormProjects/laya/mcp/laya_mcp.py
```

`uv run --script` resolve as dependências (`mcp`, `httpx`) pelo cabeçalho PEP 723 do próprio
arquivo — não precisa de virtualenv. Conferir com `claude mcp list`.

### Ferramentas

| Ferramenta | Função |
|---|---|
| `laya_predict` | Entrada principal: N perguntas tipadas sobre um state, num único forward pass |
| `laya_classify` | Atalho para "qual rótulo" — uma pergunta `choice` |
| `laya_health` | Diagnóstico: API no ar, modelo carregado, auto-start possível |
| `laya_start` | Aquece o container de propósito, antes de um lote |
| `laya_checkpoints` | Os três checkpoints, budgets e para que serve cada um |

### Auto-start

Quando a API não responde, o servidor roda `docker compose up -d laya-api` no diretório do
repositório e espera `/health` reportar `model_loaded: true`. Boot frio medido: **~107s**
com a imagem já construída (o primeiro build, com download dos pesos, é bem mais longo).
Se o container cair depois, a chamada seguinte religa e tenta de novo, uma vez.

Auto-start é ignorado — com o motivo na mensagem de erro — quando `LAYA_AUTO_START=0`,
quando `LAYA_API_URL` aponta para host remoto, quando não há `docker` no PATH ou quando não
existe `docker-compose.yml` no diretório configurado.

### Variáveis de ambiente

| Var | Padrão | Função |
|---|---|---|
| `LAYA_API_URL` | `http://localhost:8000` | Onde está a Decision API. URL remota desliga o auto-start. |
| `LAYA_AUTO_START` | `1` | `0` para nunca subir container. |
| `LAYA_COMPOSE_DIR` | raiz do repositório | Diretório com o `docker-compose.yml`. |
| `LAYA_COMPOSE_SERVICE` | `laya-api` | Serviço a subir. |
| `LAYA_START_TIMEOUT` | `900` | Segundos de espera pelo carregamento do modelo. |
| `LAYA_TIMEOUT` | `120` | Timeout HTTP por predição. |

## Testes

49 testes contra o container real (sem mock): contrato MCP, qualidade de decisão,
auto-start e triagem de requisitos.

| Documento | Conteúdo |
|---|---|
| [`tests/RESULTS.md`](tests/RESULTS.md) | Registro completo: todos os 49 testes, o que cada um verifica, resultado e duração |
| [`tests/TESTS.md`](tests/TESTS.md) | Análise por tema, defeitos encontrados, limitações |
| [`tests/SPEC-RESEARCH.md`](tests/SPEC-RESEARCH.md) | Laya como triagem de requisitos para a skill `spec-research` |

```bash
./tests/run.sh              # 44 testes rápidos (~81s), exige o container no ar
./tests/run.sh --run-slow   # + 5 que derrubam e religam o container (~4min)
```

Resumo da última medição (CPU, 2026-09-22):

| Métrica | Valor |
|---|---|
| Triagem por departamento (15 casos) | 0.867 — inglês 1.000, português 0.667 |
| Detecção de reembolso (`noul`, 6 casos) | 1.000, separação 0.738 |
| Detecção de churn (`noul`, 4 casos) | 0.750 — erra ameaça implícita |
| Roteamento de escrita (5 escritas) | 5/5 corretos |
| Latência, 1 pergunta (p50) | 172.7 ms |
| Cold start via auto-start | ~108 s |
| Roteamento de painel de requisito (PT + critérios EN) | 0.923 — com corte 0.30, 8/13 automáticos sem erro |
| Detecção de ambiguidade em requisito | reprovado (≤0.50, separação ~0) |

Cada rodada grava as métricas em `tests/reports/metrics.json`.

## Skill global

`~/.claude/skills/laya/SKILL.md` — quando usar o Laya em vez de raciocinar item a item,
como montar as perguntas tipadas, escolha de checkpoint e os limites que importam
(over-confidence, teto de opções por budget de token, `score` como primitiva mais fraca).
Carregamento manual: `/laya`.

---

## Modo script (alternativa)

Roda uma predição e sai, sem servidor HTTP.

### `Dockerfile`

```dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    HF_HOME=/home/laya/.cache/huggingface

# Usuário não-root
RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home laya \
    && mkdir -p /home/laya/.cache/huggingface \
    && chown -R laya:laya /home/laya

WORKDIR /app

# Instala o Laya + PyTorch CPU (funciona bem no M2 via Rosetta/arm64)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir laya

USER laya

# Cache do Hugging Face persistente
VOLUME /home/laya/.cache/huggingface

# Script de exemplo
COPY --chown=laya:laya app.py /app/app.py

CMD ["python", "app.py"]
```

### `app.py`

```python
import json
from laya import Router

print("Carregando Laya Router...")
router = Router(preload=True, max_loaded=1)   # max_loaded=1 economiza memória

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

print("Rodando predicção...")
result = router.predict(state, questions)

print("\n=== Resultado ===")
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### `docker-compose.yml`

```yaml
services:
  laya:
    build: .
    container_name: laya
    environment:
      - LAYA_DEVICE=cpu
      - OMP_NUM_THREADS=6          # ajuste conforme seus núcleos
      - HF_HUB_OFFLINE=0
      # - HF_TOKEN=seu_token_aqui  # só se precisar de repos privados
    volumes:
      # Cache persistente dos modelos (não baixa de novo)
      - laya-hf-cache:/home/laya/.cache/huggingface
      # Opcional: montar seu próprio script
      # - ./app.py:/app/app.py
    # Se quiser expor uma API no futuro, descomente:
    # ports:
    #   - "8080:8080"
    restart: "no"

volumes:
  laya-hf-cache:
```

### Comandos

```bash
# Build + rodar uma vez
docker compose up --build

# Ou só build
docker compose build

# Rodar de novo (já com cache)
docker compose up
```

### Predição avulsa sem editar `app.py`

```bash
docker compose run --rm laya python -c "
from laya import Router
r = Router(preload=True, max_loaded=1)
print(r.predict({'body': 'Please refund me'}, {'refund': {'type': 'noul', 'instructions': 'Is a refund requested?'}}))
"
```

---

## Notas

- `OMP_NUM_THREADS` — ajuste conforme os núcleos da máquina (padrão do modo API: 6).
- `HF_TOKEN` — só necessário para repositórios privados no Hugging Face.
- `max_loaded=1` mantém apenas um checkpoint em memória por vez.
- Container roda como usuário não-root `laya` (uid/gid 10001).
