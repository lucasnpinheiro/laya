# Bateria de testes — Laya MCP

Registro completo teste a teste, com resultado e duração de cada um, em
[RESULTS.md](RESULTS.md). Este documento é a análise por tema.

49 testes em quatro frentes: **contrato MCP** (21), **qualidade de decisão** (13),
**auto-start do container** (5, opt-in) e **triagem de requisitos para o spec-research**
(10, documentada à parte em [SPEC-RESEARCH.md](SPEC-RESEARCH.md)). Rodados contra o container real em
`localhost:8000`, sem mock — as respostas do modelo abaixo são medidas, não estimadas.

Ambiente: macOS (Darwin 27), Docker Desktop, CPU (`OMP_NUM_THREADS=6`), imagem
`python:3.11-slim-bookworm` + PyTorch CPU. Data da medição: 2026-09-22.

## Como rodar

```bash
./tests/run.sh                       # 44 testes rápidos (~81s), read-only
./tests/run.sh --run-slow            # + 5 testes que derrubam e religam o container (~4min)
./tests/run.sh tests/test_spec_research.py   # só a triagem de requisitos (~62s)
./tests/run.sh tests/test_mcp_contract.py -v
```

O runner usa `uv run --with mcp --with pytest`; não há virtualenv para manter. Os testes
rápidos exigem o container **já rodando** (`docker compose up -d`); os lentos o derrubam de
propósito e o religam ao final.

Cada rodada grava as métricas medidas em `tests/reports/metrics.json`.

## Resultado

```
44 passed, 5 skipped in 81s         # padrão
 5 passed in 229.25s                # --run-slow
```

| Arquivo | Testes | O que cobre |
|---|---|---|
| `test_mcp_contract.py` | 21 | Descoberta de ferramentas, schema, validação de entrada, propagação de erro, forma da resposta |
| `test_decision_quality.py` | 13 | Acurácia por primitiva, roteamento, determinismo, composição, latência, limites |
| `test_autostart.py` | 5 | Subida automática do container, e quando ela deve ser recusada |
| `test_spec_research.py` | 10 | Laya como triagem de requisitos: qual screening passa, qual reprova — [SPEC-RESEARCH.md](SPEC-RESEARCH.md) |

---

## 1. Contrato MCP (21 testes)

Não julgam o modelo — verificam que um cliente consegue descobrir e usar as ferramentas, que
entrada inválida morre antes de chegar na API, e que a mensagem de erro sobrevive à volta.

**Descoberta e schema (4)** — as 5 ferramentas existem; toda ferramenta tem descrição
utilizável (>40 chars); `laya_predict` declara `state`/`questions`/`model` com
`state` e `questions` obrigatórios; `laya_checkpoints` lista os três checkpoints com
`context` 512 / 1024 / 1024 e um `warning` cada.

**Validação de entrada (8)** — rejeitados localmente, sem gastar chamada na API:

| Entrada inválida | Erro esperado |
|---|---|
| `choice` sem `criteria` | cita `criteria` e o nome da pergunta |
| `choice` com uma opção só | cita "two" |
| `score` com `criteria` como objeto | cita "list" |
| `noul` com `criteria` | cita "noul" |
| `type` desconhecido (`regression`) | lista os tipos válidos |
| sem `instructions` | cita `instructions` |
| `questions` vazio | cita "non-empty" |
| `laya_classify` com 1 label | cita "two" |

**Propagação de erro (1)** — guarda de regressão. O `mcp` 2.x só repassa texto de
`ToolError`; qualquer outra exceção vira `"Error executing tool laya_predict"` e o motivo
some. O teste falha se a mensagem for essa string genérica.

**Forma da resposta (8)** — `answers` + `routing{model,repo,reason}`; `noul` em [0,1];
distribuição de `choice` somando 1.0 (±0.02); `state` aceito como string e como objeto;
N perguntas voltam N respostas numa chamada; `model="multilingual"` é respeitado;
checkpoint inexistente é recusado; `laya_start` com container no ar responde
`already_running: true` em <30s.

---

## 2. Qualidade de decisão (13 testes)

Fixtures rotuladas em `tests/fixtures/cases.json`: 15 casos de triagem (9 EN, 6 PT),
6 de reembolso, 4 de churn, 2 de urgência, 5 de roteamento.

Os limiares são **guardas de regressão abaixo do medido**, não metas. Servem para pegar
troca de checkpoint ou mudança de prompt que degrade as respostas, tolerando as fraquezas
conhecidas do modelo.

### Triagem por departamento (`choice`, 4 opções)

| Recorte | n | Acurácia | Guarda |
|---|---|---|---|
| Geral | 15 | **0.867** | ≥ 0.80 |
| Inglês | 9 | **1.000** | — |
| Português | 6 | **0.667** | ≥ 0.60 |

Os dois erros são ambos PT de **vendas**:

| Caso | Esperado | Obtido | Confiança |
|---|---|---|---|
| `pt-sales-1` "orcamento para o plano empresarial com 200 usuarios" | sales | technical | 0.363 |
| `pt-sales-2` "Quanto custa fazer upgrade do plano basico" | sales | billing | 0.803 |

Achado que importa: o segundo erro vem com **0.803 de confiança**. Confiança alta não
protege contra erro de categoria em português. Quem usar triagem PT para vendas deve
revisar, não confiar no threshold.

### Julgamentos booleanos (`noul`)

| Pergunta | n | Acurácia | Separação (média sim − média não) | Guarda |
|---|---|---|---|---|
| Pede reembolso? | 6 | **1.000** | 0.738 | ≥0.80 acc, ≥0.30 sep |
| Ameaça cancelar? | 4 | **0.750** | 0.596 | ≥0.75 |

Reembolso: positivos em 0.925 / 0.894 / 0.609 (o menor é o caso em português), negativos
todos abaixo de 0.10. Separação limpa.

Churn erra `churn-yes-2` — *"This is the third outage this month. We are seriously
considering leaving."* — com **0.382**. Ameaça explícita ("we will cancel") pontua 0.835;
ameaça implícita cai abaixo do corte. `noul` funciona melhor com intenção declarada.

### Urgência (`score`, 3 níveis)

Só a **ordenação** é afirmada — `score` é a primitiva mais fraca do modelo (SST-5: 0.372).

| Caso | Score |
|---|---|
| "Production is completely down... losing revenue every minute" | 1.95 / 2.0 |
| "Whenever you get a chance... No rush at all" | 0.454 / 2.0 |

Separação ampla, ordenação correta.

### Roteamento

Mesma intenção de reembolso em 5 escritas. O roteador acertou o checkpoint nos 5, e a
intenção sobreviveu em todos:

| Caso | Escrita | Roteado para | p(reembolso) |
|---|---|---|---|
| `route-en` | latina | english | 0.792 |
| `route-hi` | devanágari | multilingual | 0.995 |
| `route-ja` | japonesa | multilingual | 0.989 |
| `route-ar` | árabe | multilingual | 0.889 |
| `route-ru` | cirílica | multilingual | 0.974 |

É o teste que justifica o roteador existir: o checkpoint inglês colapsa nessas escritas
mantendo confiança alta.

### Comportamento

**Determinismo** — mesma chamada duas vezes devolve probabilidade idêntica (1.0 e 1.0).
Não há amostragem; resultados são reproduzíveis entre rodadas (as tabelas acima se repetiram
byte a byte em execuções separadas).

**Composição** — perguntar `department` sozinha ou junto de mais três dá resposta idêntica
(`technical`, confiança 0.4417 nas duas). As perguntas dividem o mesmo forward pass sem
contaminar umas às outras.

**`laya_classify` = `laya_predict`** — o atalho devolve label e confiança idênticos à chamada
`choice` equivalente. É açúcar sintático, não outro caminho.

**Ranking de confiança** — caso claro ("Refund the duplicate charge on invoice #4411")
pontua 0.730 em `billing`; caso vago ("Hi, I have a question about my account") pontua
0.023 em `other`. O ranking de confiança é utilizável mesmo o modelo sendo over-confident
em absoluto.

### Alta cardinalidade

4 labels vs 40 labels, mesmo texto técnico:

| Labels | Resposta | Confiança |
|---|---|---|
| 4 | technical ✓ | 0.689 |
| 40 | technical ✓ | 0.996 |

**Este resultado contraria a expectativa da documentação** — com 40 labels o modelo acertou,
e com confiança *maior*. Um caso não derruba o limite documentado (o número do Banking77 vem
de 77 labels densamente parecidas, aqui os 36 labels extras eram ruído não competitivo), mas
o teste registra a medição em vez de afirmar a degradação. O teste só exige que o caminho
documentado como bom (poucos labels) continue funcionando.

### Latência (round-trip MCP + HTTP + inferência CPU)

| Medida | Valor |
|---|---|
| 1 pergunta, p50 | **172.7 ms** |
| 1 pergunta, mín / máx | 164.5 / 280.3 ms |
| 10 perguntas em lote | 1386.4 ms |
| 10 perguntas, por pergunta | 138.6 ms |

São números de **CPU**, não os 32.8 ms de T4 do paper. O ganho de lote aqui é modesto
(138.6 vs 172.7 ms por pergunta, ~20%), bem abaixo do 2x+ relatado em GPU. Guardas: p50 <5s,
lote <10s — folgadas de propósito, para não quebrar em máquina lenta.

---

## 3. Auto-start (5 testes, `--run-slow`)

Derrubam o container de verdade com `docker compose down` e o religam ao final.

| Teste | Verifica |
|---|---|
| `test_predict_starts_the_container_when_it_is_down` | `laya_health` reporta `reachable: false`, depois `laya_predict` sobe o stack e responde |
| `test_health_reports_the_api_as_down_without_starting_it` | `laya_health` é diagnóstico: nunca sobe nada sozinho |
| `test_start_warms_the_stack_and_reports_the_cold_path` | `laya_start` com container parado reporta `already_running: false` e tempo real |
| `test_offline_hint_is_returned_when_auto_start_is_impossible` | Sem `docker-compose.yml` alcançável, o erro nomeia o motivo |
| `test_remote_api_url_never_triggers_a_local_container` | `LAYA_API_URL` remoto não faz subir container local |

**Cold start medido: 107.8 s** (e 116.9 s numa segunda execução), com a imagem já
construída e o volume de pesos populado. É o tempo de `docker compose up -d` mais o
carregamento do Router até `/health` reportar `model_loaded: true`.

---

## Bugs encontrados pelos testes

Três defeitos reais, todos corrigidos durante a bateria:

1. **Mensagens de erro engolidas.** O `mcp` 2.x só repassa texto de `ToolError`. Os
   `ValueError`/`RuntimeError` do servidor viravam `"Error executing tool laya_predict"`,
   apagando o hint de auto-start e as mensagens de validação. Trocado para `ToolError`;
   `test_error_messages_are_specific_not_generic` guarda contra a volta.

2. **`laya_start` mentia com o container derrubado.** A flag de cache `_api_ready`
   sobrevivia a um `docker compose down`, então `_ensure_api()` retornava de imediato e a
   ferramenta respondia `ready: true, elapsed_seconds: 0.0` sem nada no ar. Encontrado por
   `test_start_warms_the_stack_and_reports_the_cold_path`. Corrigido invalidando a flag
   quando o probe diz que a API não está carregada.

3. **`start_period` do healthcheck curto demais.** 60 s contra ~2 min de carregamento do
   modelo: o container ficava `unhealthy` sem motivo. Subido para 300 s.

## Limitações desta bateria

- **Amostras pequenas.** 15 casos de triagem, 6 de reembolso, 4 de churn. Servem como guarda
  de regressão e como leitura qualitativa; não são benchmark. Os números por recorte
  (ex.: 0.667 em PT) carregam intervalo de confiança largo com n=6.
- **Calibração não é medida.** Não há ECE/Brier aqui — exigiria centenas de casos rotulados.
  A documentação do modelo reporta ECE 0.466 antes de temperature fitting e 0.081 depois;
  nada nesta bateria confirma ou refuta isso.
- **Um só ambiente.** CPU, macOS, Docker Desktop. Latências em GPU ou em Linux nativo serão
  diferentes.
- **Fixtures em PT sem acentuação** nos textos (escritos em ASCII), o que pode penalizar
  levemente a detecção de idioma. Não foi isolado.
- **Sem teste de carga.** Concorrência, uso de memória sob rajada e comportamento com
  `max_loaded=1` alternando idiomas não foram exercitados.
