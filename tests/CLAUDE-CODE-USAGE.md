# Laya sobre o historico do Claude Code

Pergunta: quais decisoes que o agente repete a cada prompt do desenvolvedor a Laya segura?
Fonte: `~/.claude/history.jsonl` (6658 prompts, 4466 uteis), casos parafraseados e
anonimizados em `fixtures/claude_code_usage_cases.json` (rotulos por LLM — revisar).
Perfil do uso: `fixtures/claude_code_usage_profile.md`. Bateria:
`tests/test_claude_code_usage.py`; metricas: `reports/claude_code_usage_metrics.json`.
Medido em 2026-09-26, laya 0.3.5, CPU, 96 casos x 3 modos em ~70 s.

Modos: `pt` (texto como veio, router) · `pt_ml` (texto PT, `model=multilingual`) ·
`en` (LLM traduz antes, router). Criterios sempre em ingles.

## Resultado

| Pergunta | Tipo | pt | pt_ml | en | Veredito |
|---|---|---|---|---|---|
| `request_kind` (6 classes) | choice | 0.667 | 0.625 | **0.792** | COM PORTAO: en, conf >= 0.887 aceita 42% com precisao 1.0 |
| `work_area` (6 classes) | choice | 0.542 | 0.625 | **0.708** | COM PORTAO: en conf >= 0.719 (38%); pt_ml conf >= 0.859 (33%) |
| `contains_credentials` | noul | 0.562 | 0.500 | **0.938** | APROVADO SO EN: corte 0.70 separa por inteiro (margem 0.14) |
| `destructive_intent` | noul | 0.562 | 0.562 | 0.500 | REPROVADO: separacao 0.13 |
| `is_followup` | noul | 0.562 | 0.312 | 0.562 | REPROVADO: separacao ~0 ou invertida |

## Leitura

- **Traduzir ajuda aqui**, ao contrario de `spec_research` (onde PT + criterio EN ganhou).
  Prompt de desenvolvedor e curto, cheio de jargao e de caminho de arquivo; em PT o router
  divide entre `english` e `multilingual` e a confianca deixa de separar acerto de erro.
- **Segredo em PT nao separa**: "nao" fica em 0.80-0.96. Palavra "senha" no historico e
  quase sempre senha de fila/atendimento ou feature de tela — o modelo reage a palavra.
- **Apagar dados e continuacao falham** pelo mesmo motivo de `spec_research/ambiguity`:
  sao julgamentos sobre forma/contexto, nao sobre o assunto do texto. Ficam com
  regex/hook (DROP/DELETE/TRUNCATE/`push --force`) e heuristica de tamanho.
- Portoes foram achados na propria amostra (16-24 casos): sonda, nao benchmark.
  `test_bank_verdicts_still_hold` falha se um portao aprovado passar a errar.
