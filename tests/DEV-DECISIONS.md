# Laya + desenvolvimento de software — perguntas de decisão

Pergunta de partida: **quais decisões fechadas do dia a dia de desenvolvimento o Laya
responde bem o bastante para a LLM parar de raciocinar item a item?** Oito perguntas,
medidas em `tests/test_dev_decisions.py`, casos em `tests/fixtures/dev_decisions_cases.json`
(6–8 por pergunta, pareados PT/EN, critérios sempre em inglês). Banco pronto para uso em
`question_banks/dev_decisions.json`, servido pelo tool `laya_question_bank`.

```bash
./tests/run.sh tests/test_dev_decisions.py      # ~26s
```

**Amostra pequena.** São sondas, não benchmark: um erro a mais move a acurácia 12–17
pontos. Os cortes `noul` foram ajustados nos próprios casos — hipótese a recalibrar no seu
dado, não constante.

## Resultados (medido em 2026-09-24)

| Pergunta | Tipo | pt/en | en/en | Veredito |
|---|---|---|---|---|
| `commit_type` | choice 6 | 0.750 (gate ≥0.40: 4/8, precisão 1.0) | **1.000** | APROVADO |
| `issue_kind` | choice 4 | 0.375 | **1.000** (gate ≥0.30: 6/8) | APROVADO só em EN |
| `review_verdict` | choice 3 | 0.833 (gate ≥0.30: 3/6, precisão 1.0) | **1.000** | APROVADO |
| `security_sensitive` | noul | separa no corte 0.10 | separa no corte 0.10 | SINALIZADOR |
| `breaking_change` | noul | não separa (sim a 0.01) | separa no corte 0.10 | SINALIZADOR só em EN |
| `test_failure_cause` | choice 4 | 0.250 | 0.625 | REPROVADO |
| `log_severity` | choice 4 | 0.625 | 0.625 | REPROVADO |
| `touches_schema` | noul | separação −0.13 | sobrepõe | REPROVADO |

## Leitura

1. **Classificar o *tipo* de um texto curto funciona; diagnosticar não.** Commit, issue e
   veredito de review são rótulos sobre o que o texto *diz*. Causa de falha de teste e
   severidade de log exigem inferir consequência — quase tudo vira `product_bug`, e
   `critical` é rebaixado para `warning`, o pior erro possível para plantão.
2. **Aqui traduzir o texto compensa**, ao contrário do roteamento de requisito
   (`SPEC-RESEARCH.md`, onde pt/en venceu). `issue_kind` sai de 0.375 para 1.000 em EN.
   Regra prática: cada pergunta declara seu `_mode`; não generalize de uma para outra.
3. **`noul` sai baixo e descalibrado.** Com corte 0.5, `breaking_change` e
   `security_sensitive` acertam só metade. Mas a *ordem* é boa: nenhum "sim" abaixo de
   nenhum "não" no corte 0.10. Uso como **sinalizador** — acima do corte, manda para o
   painel; abaixo, segue o fluxo normal. Nunca como liberação ("não toca segurança").
4. **`touches_schema` é enganado por palavras.** Deu 1.0 para "corrigir o texto do e-mail
   de boas-vindas" em PT. Decidir se precisa de `.sql` continua com a LLM/`mysql`.

## Receita

1. `laya_question_bank(name="dev_decisions")` → copie o bloco `question`.
2. Mande o texto no idioma do `_mode` (traduza para EN quando `_mode = "en"`).
3. choice: aceite só com `confidence ≥ _confidence_gate`; abaixo, a LLM decide.
   noul: `≥ _noul_cut` levanta a bandeira; abaixo, nada é dispensado.
4. Empacote no mesmo `laya_predict` só perguntas do mesmo `_mode`.
