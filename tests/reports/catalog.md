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
