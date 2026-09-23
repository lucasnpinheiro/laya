# Resultados dos testes — Laya MCP

Registro completo da suíte: todos os 49 testes, o que cada um verifica, e o resultado
medido. A análise por tema está em [TESTS.md](TESTS.md); a avaliação do Laya como triagem
de requisitos está em [SPEC-RESEARCH.md](SPEC-RESEARCH.md).

## Execução documentada

| | |
|---|---|
| Data | 2026-09-22 |
| Comando | `./tests/run.sh --run-slow --junitxml=tests/reports/junit.xml` |
| Resultado | **49 de 49 passaram**, 353.9 s |
| Ambiente | macOS (Darwin 27), Docker Desktop, CPU, `OMP_NUM_THREADS=6` |
| Container | `laya-api`, imagem `python:3.11-slim-bookworm` + PyTorch CPU, usuário `laya` (uid 10001) |
| Processo | sem entrypoint; `uvicorn app:app --host 0.0.0.0 --port 8000` como PID 1 |
| Alvo | container real em `http://localhost:8000`, **sem mock** |

Os testes chamam o servidor MCP por stdio, que chama a API HTTP, que roda o modelo. Nenhuma
camada é simulada — o que está medido abaixo é o comportamento de ponta a ponta.

## Reproduzir

```bash
docker compose up -d                                  # a suíte rápida exige o container no ar
./tests/run.sh                                        # 44 rápidos (~81 s)
./tests/run.sh --run-slow                             # + 5 que derrubam e religam o container (~5 min)
./tests/run.sh --run-slow --junitxml=tests/reports/junit.xml
uv run --no-project python tests/make_catalog.py      # regenera o catálogo abaixo
```

Três arquivos de saída ficam em `tests/reports/`: `junit.xml` (execução),
`metrics.json` (qualidade de decisão) e `spec_research_metrics.json` (triagem de
requisitos). O catálogo é gerado do `junit.xml` mais os docstrings — renomear um teste
atualiza a documentação na próxima geração, em vez de deixá-la mentir.

---

## Catálogo completo

Durações são só do corpo do teste; o custo de subir o servidor MCP fica na fixture e não
aparece aqui.

<!-- CATALOG:START -->
<!-- GERADO por tests/make_catalog.py - nao editar a mao -->

49 de 49 testes passaram — 353.9s somados.

### `test_mcp_contract.py` — Contrato MCP (21 testes, 1.4s)

| Teste | Verifica | Resultado | Tempo |
|---|---|---|---|
| `test_server_exposes_the_documented_tools` | All five documented tools show up in client discovery. | passou | 0.02s |
| `test_every_tool_is_described` | Every tool carries a description a client can choose from. | passou | 0.00s |
| `test_predict_schema_advertises_its_arguments` | The laya_predict schema declares state, questions and model, the first two required. | passou | 0.00s |
| `test_health_reports_a_loaded_model` | laya_health reports the API up, the model loaded and auto-start available. | passou | 0.01s |
| `test_checkpoints_are_listed_with_their_budgets` | laya_checkpoints lists the three checkpoints with each one's context budget and warning. | passou | 0.00s |
| `test_choice_without_criteria_is_rejected` | A choice question without criteria is refused before it reaches the API. | passou | 0.00s |
| `test_choice_with_a_single_option_is_rejected` | A choice question with a single option is refused. | passou | 0.00s |
| `test_score_criteria_must_be_an_ordered_list` | A score question needs criteria as an ordered list, not an object. | passou | 0.00s |
| `test_noul_must_not_carry_criteria` | A noul question carrying criteria is refused - the type takes no options. | passou | 0.00s |
| `test_unknown_question_type_is_rejected` | An unknown question type is refused, naming the valid ones. | passou | 0.00s |
| `test_missing_instructions_is_rejected` | A question without instructions is refused. | passou | 0.00s |
| `test_empty_questions_is_rejected` | An empty questions object is refused. | passou | 0.00s |
| `test_classify_needs_at_least_two_labels` | laya_classify with a single label is refused. | passou | 0.00s |
| `test_error_messages_are_specific_not_generic` | Regression guard: mcp 2.x only forwards ToolError text, everything else is flattened. | passou | 0.05s |
| `test_predict_returns_answers_and_routing_metadata` | The result carries answers plus routing with model, repo and the reason for the pick. | passou | 0.18s |
| `test_choice_answer_carries_a_full_probability_distribution` | A choice answer carries the full distribution summing to 1.0, not just the winner. | passou | 0.17s |
| `test_a_state_object_is_accepted_alongside_a_plain_string` | state accepts an object of named fields, not only a plain string. | passou | 0.24s |
| `test_all_questions_are_answered_in_one_call` | The whole point of the model: N questions, one forward pass. | passou | 0.47s |
| `test_explicit_checkpoint_override_is_honoured` | Forcing model overrides the router's pick. | passou | 0.18s |
| `test_unknown_checkpoint_name_is_refused` | An unknown checkpoint name is refused. | passou | 0.01s |
| `test_start_is_idempotent_when_already_running` | laya_start on a running container reports already-running and restarts nothing. | passou | 0.02s |

### `test_decision_quality.py` — Qualidade de decisão (13 testes, 19.2s)

| Teste | Verifica | Resultado | Tempo |
|---|---|---|---|
| `test_department_triage_accuracy` | Department routing over 15 labelled tickets, English and Portuguese. | passou | 5.33s |
| `test_triage_works_in_portuguese_not_just_english` | Guards the router: a Latin-script non-English state must not collapse. | passou | 1.32s |
| `test_refund_detection` | noul: does the user explicitly ask for a refund - 6 labelled cases. | passou | 1.45s |
| `test_churn_risk_detection` | noul: does the user threaten to cancel - 4 labelled cases. | passou | 0.89s |
| `test_urgency_score_orders_blocking_above_trivial` | `score` is the model's weakest primitive, so this only asserts the ordering. | passou | 0.40s |
| `test_router_sends_non_latin_scripts_to_the_multilingual_checkpoint` | The English checkpoint stays confident while being wrong on these scripts. | passou | 0.56s |
| `test_non_latin_refund_intent_survives_routing` | End-to-end proof the routing is worth something: same intent, four scripts. | passou | 0.63s |
| `test_answer_is_stable_whether_asked_alone_or_alongside_others` | Extra questions share one forward pass; they must not change each other's answers. | passou | 1.28s |
| `test_repeated_calls_are_deterministic` | No sampling: the same state and questions must give byte-identical probabilities. | passou | 1.17s |
| `test_classify_matches_the_equivalent_predict_call` | laya_classify is a shortcut, not a different model path. | passou | 1.72s |
| `test_high_cardinality_choice_degrades_as_documented` | Options share a fixed token budget, so many labels blur together. | passou | 1.39s |
| `test_confidence_is_higher_on_clear_cases_than_ambiguous_ones` | Calibration sanity: the model ships over-confident, but ranking should still hold. | passou | 0.56s |
| `test_latency_stays_in_the_expected_envelope` | Round-trip through MCP + HTTP + CPU inference. Not the paper's GPU numbers. | passou | 2.51s |

### `test_autostart.py` — Auto-start do container (5 testes, 259.6s)

| Teste | Verifica | Resultado | Tempo |
|---|---|---|---|
| `test_predict_starts_the_container_when_it_is_down` | With the stack down, a prediction boots it and still answers. | passou | 136.87s |
| `test_health_reports_the_api_as_down_without_starting_it` | laya_health is a diagnostic: it must never launch anything by itself. | passou | 4.23s |
| `test_start_warms_the_stack_and_reports_the_cold_path` | laya_start on a stopped stack reports a cold start and the real elapsed time. | passou | 116.39s |
| `test_offline_hint_is_returned_when_auto_start_is_impossible` | With no compose file in reach, the failure must name the reason, not just fail. | passou | 1.38s |
| `test_remote_api_url_never_triggers_a_local_container` | A remote LAYA_API_URL must not make the server start containers on this machine. | passou | 0.74s |

### `test_spec_research.py` — Triagem de requisitos (spec-research) (10 testes, 73.7s)

| Teste | Verifica | Resultado | Tempo |
|---|---|---|---|
| `test_ambiguity_detection_across_language_modes` | Qual modo de idioma acha os requisitos vagos - recall importa mais que acuracia. | passou | 5.66s |
| `test_ambiguity_screening_recall_at_a_lower_threshold` | Uso real: filtro, nao juiz. Com corte baixo, quanto sobra para a LLM revisar? | passou | 5.57s |
| `test_requirement_versus_context_filtering` | Separar requisito de contexto historico, para a LLM nao ler o PRD inteiro linha a linha. | passou | 3.34s |
| `test_panel_routing_across_language_modes` | Roteia cada requisito para o painel de revisao - `choice` de 6 opcoes, nos 3 modos. | passou | 11.64s |
| `test_panel_routing_confidence_separates_right_from_wrong` | A confianca do roteamento de painel separa acerto de erro - e o que autoriza o portao. | passou | 0.00s |
| `test_ui_surface_detection` | Detecta superficie de UI, que dispara o gate UI_DESIGN_APPROVED. | passou | 6.52s |
| `test_unresolved_severity_classification` | 3 opcoes, amostra minima (1 por classe): serve como sonda, nao como medida. | passou | 4.01s |
| `test_full_prd_screening_cost` | Custo de varrer um PRD inteiro: 13 requisitos, 3 perguntas cada, uma chamada por requisito. | passou | 25.37s |
| `test_cost_of_packing_versus_separate_calls` | Custo de empacotar perguntas contra chamadas separadas, medido em vez de assumido. | passou | 7.81s |
| `test_confidence_gated_panel_routing_is_safe_to_automate` | A receita aprovada: PRD em portugues, criterios em ingles, corte de confianca. | passou | 3.81s |

<!-- CATALOG:END -->

---

## Resultados medidos

### Qualidade de decisão

Fixtures rotuladas em `tests/fixtures/cases.json`. Limiares nos testes são guardas de
regressão abaixo do medido, não metas.

**Triagem por departamento** (`choice`, 4 opções, 15 casos)

| Recorte | n | Acurácia |
|---|---|---|
| Geral | 15 | **0.867** |
| Inglês | 9 | **1.000** |
| Português | 6 | **0.667** |

Os dois erros são ambos PT de vendas — e o pior deles veio confiante:

| Caso | Esperado | Obtido | Confiança |
|---|---|---|---|
| `pt-sales-1` | sales | technical | 0.363 |
| `pt-sales-2` | sales | billing | **0.803** |

**Julgamentos booleanos** (`noul`)

| Pergunta | n | Acurácia | Separação |
|---|---|---|---|
| Pede reembolso? | 6 | **1.000** | 0.738 |
| Ameaça cancelar? | 4 | **0.750** | 0.596 |

Positivos de reembolso: 0.925 / 0.894 / 0.609 (o menor é o caso em português); negativos
todos abaixo de 0.10. O erro de churn é `churn-yes-2` ("we are seriously considering
leaving") em **0.382** — ameaça explícita pontua 0.835, implícita fica abaixo do corte.

**Urgência** (`score`, 3 níveis) — só a ordenação é afirmada:
produção parada 1.95 / 2.0 contra "sem pressa" 0.454 / 2.0.

**Roteamento por escrita** — 5 de 5 corretos, intenção preservada em todas:

| Caso | Escrita | Roteado | p(reembolso) |
|---|---|---|---|
| `route-en` | latina | english | 0.792 |
| `route-hi` | devanágari | multilingual | 0.995 |
| `route-ja` | japonesa | multilingual | 0.989 |
| `route-ar` | árabe | multilingual | 0.889 |
| `route-ru` | cirílica | multilingual | 0.974 |

**Comportamento** — determinismo confirmado (mesma chamada, probabilidade idêntica; as
tabelas se repetem byte a byte entre execuções). Composição: perguntar `department` sozinha
ou junto de mais três dá `technical` com confiança 0.4417 nos dois casos. `laya_classify`
devolve label e confiança idênticos ao `laya_predict` equivalente. Confiança ranqueia: caso
claro 0.730, caso vago 0.023.

**Alta cardinalidade** — 4 labels: `technical` a 0.689; 40 labels: `technical` a 0.996.
Contraria a expectativa da documentação. Um caso não derruba o limite do Banking77 (lá são
77 rótulos densamente parecidos; aqui os 36 extras eram ruído não competitivo), então o
teste registra a medição em vez de afirmar degradação.

**Latência** (round-trip MCP + HTTP + inferência CPU)

| Medida | Valor |
|---|---|
| 1 pergunta, p50 | **172.7 ms** |
| 1 pergunta, mín / máx | 164.5 / 280.3 ms |
| 10 perguntas em lote | 1386.4 ms |
| 10 perguntas, por pergunta | 138.6 ms |

São números de CPU, não os 32.8 ms de T4 do paper. Ganho de lote: ~20%, bem abaixo do 2x+
relatado em GPU.

### Auto-start

**Cold start medido: 107.8 s** (116.9 s numa segunda execução), com imagem construída e
volume de pesos populado. É `docker compose up -d` mais o carregamento do Router até
`/health` reportar `model_loaded: true`.

As recusas também são testadas: sem `docker-compose.yml` alcançável e com `LAYA_API_URL`
remoto, o erro nomeia o motivo em vez de apenas falhar.

### Triagem de requisitos (spec-research)

Cinco triagens candidatas, cada uma em três modos de idioma. Detalhe e receita em
[SPEC-RESEARCH.md](SPEC-RESEARCH.md).

**Roteamento de painel — aprovado**

| Modo | Acurácia | Confiança certo | Confiança errado |
|---|---|---|---|
| `pt/pt` | 0.615 | 0.594 | 0.250 |
| **`pt/en`** | **0.923** | 0.504 | 0.133 |
| `en/en` | 0.846 | 0.658 | 0.152 |

Texto em português com critérios em inglês ganha de traduzir o PRD inteiro. Com corte de
confiança em 0.30: **8 de 13 roteados sozinhos com precisão 1.000**, 5 escalados.

**Detecção de ambiguidade — reprovado**

| Modo | Acurácia | Recall nos vagos | Separação |
|---|---|---|---|
| `pt/pt` | 0.500 | 0.600 | **−0.11** |
| `pt/en` | 0.100 | 0.000 | **−0.42** |
| `en/en` | 0.500 | 0.000 | +0.06 |

Separação negativa: deu probabilidade maior de "vago" aos requisitos concretos.

**Demais** — severidade de pendência 0.333 (acaso, 3 classes); superfície de UI com recall
0.5 nos modos precisos; requisito-vs-contexto 0.833 em inglês e 0.667 em português.
Nenhuma entra.

**Custo** — varrer 13 requisitos com 3 perguntas cada: 14.9 s, ~1.1 s por requisito, zero
token de LLM. Empacotar perguntas **não economiza em CPU**:

| Medida | Tempo |
|---|---|
| `noul` (ambiguidade) | 225.9 ms |
| `noul` (superfície de UI) | 209.7 ms |
| `choice` com 6 opções | 653.8 ms |
| As 3 separadas | 1089.4 ms |
| As 3 empacotadas | 1061.6 ms |

Ganho de empacotar: 2.6%. Uma `choice` de 6 opções custa 2.89× um `noul`.

---

## Defeitos encontrados pelos testes

Quatro, todos corrigidos e cobertos por teste de regressão.

| # | Defeito | Como apareceu | Correção |
|---|---|---|---|
| 1 | Mensagens de erro engolidas — `mcp` 2.x só repassa texto de `ToolError`; `ValueError`/`RuntimeError` viravam `"Error executing tool laya_predict"`, apagando o hint de auto-start e as validações | Teste de propagação de erro | Todas as exceções do servidor passaram a ser `ToolError`; `test_error_messages_are_specific_not_generic` guarda |
| 2 | `laya_start` mentia com o container derrubado: a flag `_api_ready` sobrevivia ao `docker compose down`, então respondia `ready: true, elapsed_seconds: 0.0` sem nada no ar | `test_start_warms_the_stack_and_reports_the_cold_path` | Flag invalidada quando o probe diz que o modelo não está carregado |
| 3 | `start_period` do healthcheck em 60 s contra ~2 min de carregamento — container ficava `unhealthy` sem motivo | Observado no primeiro boot | Subido para 300 s |
| 4 | Conclusão errada sobre empacotamento de perguntas: o primeiro teste comparava 1 pergunta com 3 e "provava" que empacotar era ruim | Revisão do próprio teste | Refeito comparando as mesmas 3 perguntas empacotadas vs separadas; passou a medir em vez de afirmar |

## Limitações desta suíte

- **Amostras pequenas.** 15 casos de triagem, 6 de reembolso, 4 de churn, 13 de roteamento
  de painel, 3 de severidade. Servem como guarda de regressão e leitura qualitativa, não
  como benchmark — os números por recorte carregam intervalo largo.
- **Calibração não é medida.** Sem ECE/Brier; exigiria centenas de casos rotulados. O
  material do modelo reporta ECE 0.466 antes de temperature fitting e 0.081 depois, e nada
  aqui confirma ou refuta isso.
- **Um só ambiente.** CPU, macOS, Docker Desktop. GPU ou Linux nativo darão outras latências.
- **Fixtures escritas para o teste**, não extraídas de PRDs ou tickets reais. Textos em PT
  sem acentuação (ASCII), o que pode penalizar a detecção de idioma — não foi isolado.
- **Um requisito por chamada** na triagem de requisitos; PRD real mistura vários assuntos
  por parágrafo.
- **Checkpoint `typed-decisions` não exercitado** — é o afinado em workflows tipados, e
  pode mudar o teto da triagem de requisitos.
- **Sem teste de carga.** Concorrência, memória sob rajada e alternância de idioma com
  `max_loaded=1` não foram exercitados.
