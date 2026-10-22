# Laya — System 1 Decision Model (Docker)

Rodar o modelo [Laya](https://huggingface.co/convaiinnovations/laya) em container, via CPU,
exposto como API HTTP (FastAPI) ou como script único.

O repositório está configurado no **modo API**. O modo script está documentado abaixo caso
você prefira rodar um batch isolado.

## Estrutura

| Arquivo | Função |
|---|---|
| `Dockerfile` | Python 3.11 slim + PyTorch CPU + `laya==0.3.5` (pin = versão rodando no container) + FastAPI/uvicorn, usuário não-root `laya` (uid 10001) |
| `app.py` | API FastAPI com `/health` e `/predict` |
| `docker-compose.yml` | Serviço `laya-api` na porta 8000 + volume de cache do Hugging Face |
| `mcp/laya_mcp.py` | Servidor MCP para Claude Code, com auto-start do container |
| `tests/` | Bateria de testes + documentação dos resultados medidos |
| `question_banks/spec_research.json` | Perguntas validadas para triagem de requisitos, com vereditos de aprovação/reprovação |
| `question_banks/model_routing.json` | `effort_level` do roteador de modelos do `~/.agents`: só rebaixa `medium`→`low` (p(low) ≥ 0.60), nunca sobe |

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

### Desempenho: warm-up, health e latência

- **Warm-up no boot**: após carregar o `Router`, o `app.py` faz uma predição descartável no checkpoint
  `LAYA_WARMUP` (**padrão desligado**; ex. `multilingual`). Antes, a 1a chamada
  real levava >20 s (carga lazy do checkpoint); depois, ms. `max_loaded=1`: trocar de checkpoint recarrega.
- **Achado**: carregar um checkpoint no container (`mem_limit: 4g`; testado também 6g com `english`) foi morto por OOM (exit 137,
  `OOMKilled=true`) e `restart: unless-stopped` entra em loop; por isso o warm-up ficou opt-in. Investigar antes de ligar
  (VM do Docker de 8 GB dividida com outros stacks).
- `GET /health` agora devolve também `warmed: true|false`.
- `/predict` é `def` (threadpool) com lock: `/health` e o healthcheck do Docker não travam durante a inferência.
- Toda resposta traz o header `X-Latency-Ms`.
- Hook `~/.agents/hooks/model-router.py`: se o Laya recusa conexão (container inexistente/parado), cai no
  fallback na hora e dispara `docker compose up -d laya-api` destacado (no máximo 1x/10 min) para o próximo prompt.
  Causa do `laya indisponível (URLError)`: o container `laya-api` não existia (removido), e só o MCP tinha auto-start.

### Latência do /predict (2026-10-22, medido com 20 chamadas, host carregado por outros stacks)

Onde o tempo vai (encoder fp16, 6 threads): tokenização 7-17 ms; encoder ~95%; heads ~0,2 s.
O custo cresce com o número de perguntas porque cada uma vira uma sequência do batch (padding até a mais longa):

| Perguntas | p50 | p90 |
|---|---|---|
| 1 (level) | 1,09 s | 1,33 s |
| 2 (level+risk) | 1,54 s | 1,88 s |
| 3 (level+risk+area) | 4,6 s | 7,4 s |

- Threads (fp16, 3 perguntas, container isolado): 6 -> 3,2-3,9 s; 4 -> 4,3 s; 10 -> 5,1 s; 2 -> pior. Ficou `OMP_NUM_THREADS=6`/`cpus: 6`.
- bf16 (conversão explícita do encoder): inviável no torch CPU aarch64 (minutos por chamada). fp32 não cabe em 2 GB e não ficou estável em 3 GB nos testes; fica fp16.
- O resultado não depende do tamanho do batch: level/risk idênticos (30/30, delta de risco 0,0) com 2 ou 3 perguntas, e o mesmo prompt dá probabilidades idênticas a cada chamada (sem dropout).
- Hook: faz `level+risk` (1 chamada) sempre e `area` só quando o prompt cita um tema de skill. p50 do hook: 0,96 s sem a chamada de `area` (antes 3,3-4,5 s), 2,86 s com ela; fallback com porta morta 0,09 s.
- Variação de tier no mesmo prompt: não é não-determinismo do modelo; é a cota 50/35/15 (`tier_quota.py`) que desloca rapido -> padrao conforme o histórico.

### Docker Desktop recomendado

Settings > Resources: 8 CPUs, ~10 GB de memória, 4 GB de swap; VMM com Apple Virtualization (VZ) e VirtioFS;
Resource Saver DESLIGADO; "Start Docker Desktop when you sign in" LIGADO. O compose usa `mem_limit: 3g`
(com 2 g os pesos mmap sofrem thrash de page cache sob carga), `mem_reservation: 1500m`, `memswap_limit: 4g`,
`cpus: 6`, `cpu_shares: 2048`, `oom_score_adj: -500` e porta em `127.0.0.1:8000` (hook e MCP usam localhost).

### Variáveis de ambiente de memória e warm-up

- `LAYA_LOWMEM` (padrão `1`): encoder em fp16 via mmap (`lowmem.py`), sem init aleatório nem 2a cópia dos pesos;
  heads em fp32. `0` volta ao load antigo em fp32 e exige ~2,4 GB livres.
- `LAYA_KEEPALIVE` (padrão `60`, segundos): relê as páginas mmap para mantê-las quentes no page cache.
- `LAYA_WARMUP`: no `docker-compose.yml` fica `english`; healthcheck exige `warmed: true`.
- Causa do OOM: VM do Docker Desktop (7,65 GB) esgotada pelos stacks ecom*, com o checkpoint carregando em fp32
  com cópia dupla (pico 1,8-2,4 GB); com lowmem o pico anônimo cai para ~0,5 GB.

### Evidências externas (2026-10-22)

- `wangmiaozero/laya-router-skill` v0.2.0 (3 estrelas): skill advisory com `laya_decide`; seis respostas
  (`task_type`, `complexity` 0-3, `needs_strong_reasoning`, `needs_tools`, `security_sensitive`, `risk`);
  falha = `fail_open`; MCP mantém checkpoints carregados, CLI recarrega por chamada; sem benchmarks de acurácia
  ("confiança de roteamento não é probabilidade medida"). Já coberto aqui por `model_routing`/`task_tier` + fallback.
- `skillsdirectory.com/skills/jzkk720-laya`: cita laya 0.3.4; 3 checkpoints; `lang=` quando o idioma é conhecido
  (detector só reconhece 7 idiomas latinos); acurácia 93-96% em tópico/spam, 35-65% em emoção, prompt-injection e
  estrelas; latência 35-66 ms/pergunta. Confirma as regras já na skill (gate de confiança, escalar abaixo do corte).
- `aayushch/laya`: **outro produto** (central de notificações Tauri/Svelte com LLMs), sem relação com o modelo; ignorado.
- PyPI: `laya` 0.4.13 existe (pin atual `0.3.5`). **Não atualizado**: não verificado contra `app.py`/bancos medidos;
  subir exige rebuild + reexecutar `tests/` (as medições dos bancos valem para 0.3.5).

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
  "$(which uv)" -- run --script <caminho-do-repo>/mcp/laya_mcp.py
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
./tests/run.sh              # tier fast: contrato + stubs + 1 predição (~7s), exige o container warmed
./tests/run.sh --model      # + tier model: predições reais (~6 min)
./tests/run.sh --bench      # + tier bench: medição completa dos bancos de perguntas (~10 min)
./tests/run.sh --all        # fast + model + bench (tudo menos slow)
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

As métricas só são gravadas em `tests/reports/*_metrics.json` com `--bench --write-reports` (ou `LAYA_WRITE_REPORTS=1`); predições idênticas são memoizadas na sessão (`LAYA_TEST_DISK_CACHE=1` reaproveita entre rodadas, só para desenvolvimento).

## Contribuindo / proteção contra vazamento

Os hooks em `.githooks/` bloqueiam commits com segredos (tokens, chaves, senhas em claro),
caminhos pessoais, e-mail pessoal, links de sessão, arquivos `.pem`/`.env`/`.sql` e
commits feitos com identidade diferente da do projeto.

```bash
./scripts/setup-hooks.sh                                  # ativa core.hooksPath=.githooks
cp .githooks/denylist.local.example .githooks/denylist.local   # termos sensíveis locais, não versionados
```

Acrescente em `.githooks/denylist.local` os termos que nunca podem entrar no repositório
(nomes de clientes, projetos internos). O arquivo está no `.gitignore`.
Testes do guard: `tests/run.sh tests/test_git_hooks.py`.

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
