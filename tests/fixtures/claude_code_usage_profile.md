# Claude Code usage profile (fixture justification)

Source: `~/.claude/history.jsonl`, 6658 prompts scanned, 4466 usable after dropping
bare slash commands and `[Pasted text]`/`[Image]`-only rows. Mostly pt-BR. No secrets reproduced.

## Top projects (prompts)
coneman 1900 · choconfest 1162 · belasartes 1019 · cory 747 · .agents 447 ·
raicrom 204 · gerfin 183 · gwp 120 · choconfest-ecommerce 101 · others < 75 each.

## Top slash commands
/new 932 · /spec-research 719 · /generate-spec 531 · /model 212 · /effort 196 ·
/skill-manager 123 · /obsidian-skills 69 · /usage 39 · /spec-implement 20 · /init 18.
Spec pipeline commands (spec-research + generate-spec + spec-implement) = 1270 rows,
which justifies a `plan_spec` class.

## Follow-ups (heuristic: < 60 chars and starts with executar/sim/continuar/ok/pode/aplique...)
749 of 4466 usable prompts (~17%). `executar` alone appears 524 times, `sim` 74,
`continuar`/`continue` 80. Prompts under 40 chars: 1102 (~25%).

## Credential-like content (count only)
115 prompts (display or short pasted text) match a credential pattern
(`senha:`, `password=`, `token`, `*_PASSWORD=`, `user:pass@host`, key prefixes).
~2.6% of usable prompts: rare but real, mostly browser-test logins and `.env` lines.
Hard negatives are common: "senha" is also a queue ticket (senha de atendimento)
and a feature (password confirmation on delete).

## request_kind share (first keyword match, rough)
| kind | prompts | share |
|---|---|---|
| fix_bug (erro/corrig/exception/falha) | 919 | 21% |
| run_tests (teste/navegador/playwright/validar) | 721 | 16% |
| plan_spec (spec/prd/plano) | 631 | 14% |
| new_feature (criar/adicionar/tela/relatorio/campo) | 513 | 11% |
| docs_tooling (skill/AGENTS.md/hook/mcp/instalar) | 258 | 6% |
| question (?/como/qual/por que) | 68 | 2% |
| unmatched (mostly executar/sim/paths only) | 1356 | 30% |
## Destructive wording
122 prompts use remover/apagar/deletar/drop/truncate/limpar/excluir, most of them about
UI elements or code (remove a button, a field, a validation), not data. That motivates the
hard negatives in `destructive_intent`.
