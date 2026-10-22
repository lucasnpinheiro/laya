# Laya + spec-research — o que foi testado e o que sobrou

Pergunta de partida: **a LLM junto com o Laya consegue levantar requisitos com menos
esforço?** A resposta medida é *sim, num ponto só* — e a bateria existe justamente para
separar esse ponto dos quatro onde o Laya não ajuda.

10 testes em `tests/test_spec_research.py`, fixtures em
`tests/fixtures/spec_research_cases.json` (13 requisitos de ERP/CakePHP, 10 casos de
ambiguidade, 6 de requisito-vs-contexto, 3 de pendência — todos pareados PT/EN).
Métricas em `tests/reports/spec_research_metrics.json`.

```bash
./tests/run.sh tests/test_spec_research.py      # ~62s
```

## A tese testada

O pipeline do `spec-research` repete, item a item, julgamentos tipados: *este requisito é
ambíguo? qual painel obrigatório revisa? isto é requisito ou histórico? tem superfície de
UI? esta pendência bloqueia?* Cada um é uma pergunta fechada sobre um texto curto — o
formato exato do Laya. Se ele acertar, a LLM lê só o que foi marcado, em vez de raciocinar
sobre cada linha do PRD.

Isso só vale se o Laya de fato acertar. Foi o que a bateria foi medir.

## Os três modos de idioma

Os PRDs são em português, e a medição anterior (`tests/TESTS.md`) já mostrou queda em PT.
Então cada triagem foi medida em três modos, para saber onde vale traduzir:

| Modo | Texto | Critérios | Custo da tradução |
|---|---|---|---|
| `pt/pt` | português | português | nenhum |
| `pt/en` | português | inglês | os critérios são escritos uma vez |
| `en/en` | inglês | inglês | a LLM traduz o PRD inteiro |

## Resultados

### Roteamento de painel — **aprovado**

Qual especialidade revisa o requisito (6 opções: security, database, performance, frontend,
erp_domain, code_quality). Alimenta o gate `SKILLS_RESOLVED`.

| Modo | Acurácia | Confiança quando acerta | Quando erra |
|---|---|---|---|
| `pt/pt` | 0.615 | 0.594 | 0.250 |
| **`pt/en`** | **0.923** | 0.504 | 0.133 |
| `en/en` | 0.846 | 0.658 | 0.152 |

Dois achados:

1. **Texto em português com critérios em inglês ganha de traduzir o PRD.** 0.923 contra
   0.846 do `en/en` e 0.615 do `pt/pt`. Os critérios são escritos uma vez, pelo autor da
   skill; o PRD não precisa ser tocado. É o modo mais barato *e* o mais preciso.
2. **A confiança separa acerto de erro.** A diferença entre média-quando-certo e
   média-quando-errado é 0.34 / 0.37 / 0.51 conforme o modo. Diferente da triagem de
   suporte medida em `TESTS.md` (onde um erro veio com 0.80), aqui a confiança é utilizável
   como portão.

Daí a receita, medida ponta a ponta em `test_confidence_gated_panel_routing_is_safe_to_automate`:

| Corte | Aceitos sozinhos | Escalados para a LLM | Precisão no que foi aceito |
|---|---|---|---|
| sem corte | 13 | 0 | 0.923 (erra `erp-01`) |
| **≥ 0.30** | **8** | **5** | **1.000** |
| ≥ 0.50 | 6 | 7 | 1.000 |

Com corte em 0.30, **8 dos 13 requisitos são roteados sem erro nenhum** e 5 vão para a LLM
ler. O requisito que o modelo erraria (`erp-01`, cancelamento de pedido faturado) cai
exatamente na faixa escalada.

### Detecção de ambiguidade — **reprovado**

O gate "zero silent ambiguity" é o mais atraente para automatizar, e é onde o Laya falha
pior:

| Modo | Acurácia | Recall nos vagos | Separação (vago − concreto) |
|---|---|---|---|
| `pt/pt` | 0.500 | 0.600 | **−0.11** |
| `pt/en` | 0.100 | 0.000 | **−0.42** |
| `en/en` | 0.500 | 0.000 | +0.06 |

Separação negativa significa que o modelo deu probabilidade *maior* de "vago" aos requisitos
concretos. Em `en/en` ele não marcou nenhum dos 5 requisitos vagos. O único corte que
alcança recall total (`pt/pt` em 0.1) marca 9 dos 10 requisitos — o mesmo que não filtrar.

Provável causa: "vago" é um julgamento sobre a *forma* do enunciado, não sobre o assunto —
e o modelo foi treinado para classificar conteúdo. Não é um ajuste de prompt; é a tarefa
errada. **Ambiguidade continua sendo trabalho de LLM.**

### Pendências, superfície de UI, requisito-vs-contexto — **reprovados**

| Triagem | Melhor resultado | Por que não entra |
|---|---|---|
| Severidade de pendência (`blocks` / `blocks_go_live` / `open`) | 0.333 em 3 classes | Acaso. n=3, então é sonda, não medida — mas não há sinal |
| Superfície de UI | recall 1.0 só em `pt/pt`, com acurácia 0.615 | Marca quase tudo; nos modos precisos o recall cai a 0.5, e pular `UI_DESIGN_APPROVED` é caro |
| É requisito ou contexto? | 0.833 (`en/en`), 0.667 em PT | n=6. Descartar linha de PRD por engano custa mais do que ler |

### Custo

Varredura completa do PRD de 13 requisitos, 3 perguntas cada (39 perguntas):
**14.9 s**, ou **1.14 s por requisito**, zero token de LLM.

Empacotar perguntas **não economiza em CPU**, contrariando o material do modelo:

| Medida | Tempo |
|---|---|
| `noul` (ambiguidade) | 225.9 ms |
| `noul` (superfície de UI) | 209.7 ms |
| `choice` com 6 opções (painel) | 653.8 ms |
| As 3 em chamadas separadas | 1089.4 ms |
| As 3 empacotadas numa chamada | 1061.6 ms |

Ganho de empacotar: **2.6%** — ruído. E uma pergunta `choice` de 6 opções custa **2.89×** um
`noul`: o orçamento de tokens das opções domina o custo. Em GPU o quadro do paper
provavelmente se sustenta; nesta máquina, não. Consequência prática: não vale inflar o
pacote de perguntas "porque é de graça" — não é.

## A receita

```text
PRD (português)
  │
  ├─ Laya: roteamento de painel, critérios em inglês, corte 0.30
  │    ├─ confiança ≥ 0.30 → painel atribuído direto        (~60% dos requisitos, 0 erro)
  │    └─ confiança < 0.30 → a LLM lê e decide              (~40%)
  │
  └─ LLM, sem Laya: ambiguidade · severidade de pendência · gate de UI ·
                    separar requisito de contexto · tudo que o pipeline
                    já faz (gap-analysis, prior-art, rastreabilidade)
```

O banco de perguntas validado está em
[`question_banks/spec_research.json`](../question_banks/spec_research.json), com os vereditos
de aprovação e reprovação embutidos, para não ser reaproveitado fora do que foi medido.

## Ganho honesto

O ganho é **estreito**: uma das cinco triagens candidatas passou. Ela tira da LLM o
roteamento de cerca de 60% dos requisitos por ~1 s de CPU cada, com precisão medida de
1.000 nesse recorte — e, mais importante, a bateria evita que as outras quatro sejam
automatizadas por otimismo. As quatro reprovadas são exatamente as que pareciam mais
atraentes no papel.

## Limitações

- **Amostras pequenas**: 13 requisitos para painel, 10 para ambiguidade, 6 e 3 nas demais.
  Servem para decidir *se vale tentar* e como guarda de regressão, não como benchmark. Os
  números por classe carregam intervalo largo.
- **Fixtures escritas para o teste**, não extraídas de PRDs reais do projeto. Os requisitos
  imitam o domínio (ERP/CakePHP), mas um PRD real é mais longo, mais bagunçado e mistura
  vários assuntos por parágrafo — o que provavelmente derruba o roteamento.
- **Textos em PT sem acentuação** (ASCII). Pode penalizar a detecção de idioma; não foi
  isolado.
- **Um requisito por chamada.** Não foi testado mandar um parágrafo inteiro com vários
  requisitos misturados, que é o formato real de PRD.
- **Sem checkpoint `typed-decisions`.** Ele é afinado justamente em workflows tipados; os
  testes usaram só o roteamento padrão. Vale medir antes de concluir que o teto é esse.
- **Nada aqui mede o pipeline do `spec-research` em si** — só a triagem isolada que
  entraria nele.
