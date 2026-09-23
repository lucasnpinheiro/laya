"""Laya como triagem de requisitos para a skill spec-research.

O pipeline do spec-research repete, por item, julgamentos que sao caros de fazer com LLM em
lote: este requisito e ambiguo? qual painel (security/mysql/performance/...) revisa? isto e
requisito ou contexto? tem superficie de UI? Sao perguntas tipadas sobre textos curtos, o
formato exato do Laya.

Esta bateria mede se ele realmente serve para isso, em tres modos:

  pt/pt  texto em portugues, criterios em portugues   (o PRD como veio)
  pt/en  texto em portugues, criterios em ingles      (traducao so dos criterios - barata)
  en/en  texto e criterios em ingles                  (a LLM traduz o PRD antes)

O modo mais barato que aguentar o uso e o que deve entrar na skill. Metricas vao para
tests/reports/spec_research_metrics.json.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest

REPORT_DIR = Path(__file__).parent / "reports"
_metrics: dict = {}


@pytest.fixture(scope="session", autouse=True)
def spec_report():
    yield _metrics
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "spec_research_metrics.json").write_text(
        json.dumps(_metrics, indent=2, ensure_ascii=False)
    )


def record(name: str, value) -> None:
    _metrics[name] = value


@pytest.fixture(scope="session")
def spec_cases():
    with open(Path(__file__).parent / "fixtures" / "spec_research_cases.json") as fh:
        return json.load(fh)


# --- perguntas do pipeline, nos dois idiomas -------------------------------------------

AMBIGUITY_Q = {
    "pt": "O requisito e vago ou subjetivo, sem valor concreto que permita verificar se foi atendido?",
    "en": "Is this requirement vague or subjective, with no concrete value that would let you verify it?",
}
IS_REQUIREMENT_Q = {
    "pt": "O texto declara algo que o sistema deve fazer, e nao apenas contexto ou historico?",
    "en": "Does this text state something the system must do, rather than background or history?",
}
UI_SURFACE_Q = {
    "pt": "O requisito envolve uma tela, formulario ou outro elemento visivel ao usuario?",
    "en": "Does this requirement involve a screen, form or other user-visible surface?",
}
PANEL_Q = {
    "pt": "Qual especialidade deve revisar este requisito?",
    "en": "Which specialty should review this requirement?",
}
SEVERITY_Q = {
    "pt": "Qual o impacto desta pendencia?",
    "en": "What is the impact of this open item?",
}
SEVERITY_CRITERIA = {
    "pt": {
        "blocks": "impede escrever a especificacao agora, falta informacao essencial",
        "blocks_go_live": "da para especificar, mas impede a entrada em producao",
        "open": "detalhe menor, nao impede especificar nem entrar em producao",
    },
    "en": {
        "blocks": "prevents writing the spec right now, essential information is missing",
        "blocks_go_live": "can be specified, but prevents going live",
        "open": "minor detail, blocks neither the spec nor go-live",
    },
}


def noul(laya, text: str, question: str) -> float:
    result = laya.call(
        "laya_predict",
        state=text,
        questions={"q": {"type": "noul", "instructions": question}},
    )
    return result["answers"]["q"]["noul"]


def choose(laya, text: str, question: str, criteria: dict) -> tuple[str, float]:
    result = laya.call(
        "laya_predict",
        state=text,
        questions={"q": {"type": "choice", "instructions": question, "criteria": criteria}},
    )
    answer = result["answers"]["q"]
    return answer["choice"], answer["confidence"]


MODES = [("pt", "pt"), ("pt", "en"), ("en", "en")]


def mode_name(text_lang: str, question_lang: str) -> str:
    return f"{text_lang}/{question_lang}"


# --- 1. deteccao de ambiguidade (gate "zero silent ambiguity") --------------------------

def test_ambiguity_detection_across_language_modes(laya, spec_cases):
    """Qual modo de idioma acha os requisitos vagos - recall importa mais que acuracia.

    Deixar passar um requisito ambiguo e o erro caro do pipeline.
    """
    summary = {}
    for text_lang, q_lang in MODES:
        rows = []
        for case in spec_cases["ambiguity"]:
            probability = noul(laya, case[text_lang], AMBIGUITY_Q[q_lang])
            rows.append({
                "id": case["id"],
                "expected": case["ambiguous"],
                "probability": round(probability, 4),
                "correct": (probability >= 0.5) == case["ambiguous"],
            })

        positives = [r for r in rows if r["expected"]]
        negatives = [r for r in rows if not r["expected"]]
        summary[mode_name(text_lang, q_lang)] = {
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
            "recall_on_ambiguous": round(sum(r["correct"] for r in positives) / len(positives), 3),
            "specificity_on_clear": round(sum(r["correct"] for r in negatives) / len(negatives), 3),
            "separation": round(
                statistics.mean(r["probability"] for r in positives)
                - statistics.mean(r["probability"] for r in negatives), 3),
            "cases": rows,
        }

    record("ambiguity", summary)
    best = max(summary.values(), key=lambda s: s["recall_on_ambiguous"])
    assert best["recall_on_ambiguous"] >= 0.60, (
        f"nenhum modo pega ambiguidade de forma utilizavel: "
        f"{ {k: v['recall_on_ambiguous'] for k, v in summary.items()} }"
    )


def test_ambiguity_screening_recall_at_a_lower_threshold(laya, spec_cases):
    """Uso real: filtro, nao juiz. Com corte baixo, quanto sobra para a LLM revisar?"""
    modes = {}
    for text_lang, q_lang in MODES:
        probabilities = {
            case["id"]: (noul(laya, case[text_lang], AMBIGUITY_Q[q_lang]), case["ambiguous"])
            for case in spec_cases["ambiguity"]
        }
        for threshold in (0.5, 0.3, 0.2, 0.1):
            flagged = [cid for cid, (p, _) in probabilities.items() if p >= threshold]
            caught = [cid for cid, (p, amb) in probabilities.items() if p >= threshold and amb]
            total_ambiguous = sum(1 for _, amb in probabilities.values() if amb)
            modes.setdefault(mode_name(text_lang, q_lang), {})[str(threshold)] = {
                "flagged_for_llm_review": len(flagged),
                "of_total": len(probabilities),
                "ambiguous_caught": len(caught),
                "ambiguous_total": total_ambiguous,
                "recall": round(len(caught) / total_ambiguous, 3),
            }

    record("ambiguity_thresholds", modes)
    # Um filtro so vale se existir algum corte que pegue tudo sem mandar tudo para revisao.
    usable = [
        (mode, threshold, stats)
        for mode, thresholds in modes.items()
        for threshold, stats in thresholds.items()
        if stats["recall"] == 1.0 and stats["flagged_for_llm_review"] < stats["of_total"]
    ]
    record("ambiguity_usable_filters", [{"mode": m, "threshold": t, **s} for m, t, s in usable])
    assert usable, "nenhum corte alcanca recall total sem mandar o PRD inteiro para revisao"


# --- 2. requisito vs contexto (limpeza de intake) ---------------------------------------

def test_requirement_versus_context_filtering(laya, spec_cases):
    """Separar requisito de contexto historico, para a LLM nao ler o PRD inteiro linha a linha."""
    summary = {}
    for text_lang, q_lang in MODES:
        rows = []
        for case in spec_cases["is_requirement"]:
            probability = noul(laya, case[text_lang], IS_REQUIREMENT_Q[q_lang])
            rows.append({
                "id": case["id"],
                "expected": case["is_requirement"],
                "probability": round(probability, 4),
                "correct": (probability >= 0.5) == case["is_requirement"],
            })
        summary[mode_name(text_lang, q_lang)] = {
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
            "cases": rows,
        }

    record("is_requirement", summary)
    best = max(s["accuracy"] for s in summary.values())
    assert best >= 0.80, f"separar requisito de contexto nao funciona em nenhum modo: {summary}"


# --- 3. roteamento para os paineis obrigatorios ------------------------------------------

def test_panel_routing_across_language_modes(laya, spec_cases):
    """Roteia cada requisito para o painel de revisao - `choice` de 6 opcoes, nos 3 modos.

    Alimenta o gate SKILLS_RESOLVED, que exige um veredito por painel.
    """
    summary = {}
    for text_lang, q_lang in MODES:
        criteria = spec_cases[f"panel_criteria_{q_lang}"]
        rows = []
        for case in spec_cases["requirements"]:
            choice, confidence = choose(laya, case[text_lang], PANEL_Q[q_lang], criteria)
            rows.append({
                "id": case["id"],
                "expected": case["panel"],
                "got": choice,
                "correct": choice == case["panel"],
                "confidence": round(confidence, 4),
            })
        correct = [r for r in rows if r["correct"]]
        wrong = [r for r in rows if not r["correct"]]
        summary[mode_name(text_lang, q_lang)] = {
            "accuracy": round(len(correct) / len(rows), 3),
            "mean_confidence_when_right": round(
                statistics.mean(r["confidence"] for r in correct), 3) if correct else None,
            "mean_confidence_when_wrong": round(
                statistics.mean(r["confidence"] for r in wrong), 3) if wrong else None,
            "cases": rows,
        }

    record("panel_routing", summary)
    best_mode, best = max(summary.items(), key=lambda kv: kv[1]["accuracy"])
    record("panel_routing_best_mode", {"mode": best_mode, "accuracy": best["accuracy"]})
    assert best["accuracy"] >= 0.50, (
        "roteamento de painel inutil em todos os modos: "
        f"{ {k: v['accuracy'] for k, v in summary.items()} }"
    )


def test_panel_routing_confidence_separates_right_from_wrong(laya, spec_cases):
    """A confianca do roteamento de painel separa acerto de erro - e o que autoriza o portao.

    Diferente da triagem de suporte medida em TESTS.md, onde um erro veio a 0.80 de
    confianca. Aqui a diferenca entre media-quando-certo e media-quando-errado e positiva
    nos tres modos, o que torna o corte por confianca utilizavel.
    """
    routing = _metrics.get("panel_routing")
    if routing is None:
        pytest.skip("depende de test_panel_routing_across_language_modes")

    gaps = {}
    for mode, stats in routing.items():
        right, wrong = stats["mean_confidence_when_right"], stats["mean_confidence_when_wrong"]
        gaps[mode] = None if (right is None or wrong is None) else round(right - wrong, 3)

    record("panel_confidence_gap", gaps)
    measured = {mode: gap for mode, gap in gaps.items() if gap is not None}
    assert measured, "sem dados de confianca"
    assert all(gap > 0 for gap in measured.values()), (
        f"a confianca deixou de separar acerto de erro: {measured}"
    )


# --- 4. gatilho de UI (gate UI_DESIGN_APPROVED) -------------------------------------------

def test_ui_surface_detection(laya, spec_cases):
    """Detecta superficie de UI, que dispara o gate UI_DESIGN_APPROVED.

    Errar para menos pula um gate obrigatorio, entao recall e o que importa.
    """
    summary = {}
    for text_lang, q_lang in MODES:
        rows = []
        for case in spec_cases["requirements"]:
            probability = noul(laya, case[text_lang], UI_SURFACE_Q[q_lang])
            rows.append({
                "id": case["id"],
                "expected": case["ui_surface"],
                "probability": round(probability, 4),
                "correct": (probability >= 0.5) == case["ui_surface"],
            })
        positives = [r for r in rows if r["expected"]]
        summary[mode_name(text_lang, q_lang)] = {
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
            "recall_on_ui": round(sum(r["correct"] for r in positives) / len(positives), 3),
            "cases": rows,
        }

    record("ui_surface", summary)
    best = max(s["recall_on_ui"] for s in summary.values())
    assert best >= 0.50, f"deteccao de superficie de UI inutil: {summary}"


# --- 5. severidade das pendencias (secao ## Unresolved) -----------------------------------

def test_unresolved_severity_classification(laya, spec_cases):
    """3 opcoes, amostra minima (1 por classe): serve como sonda, nao como medida."""
    summary = {}
    for text_lang, q_lang in MODES:
        rows = []
        for case in spec_cases["unresolved"]:
            choice, confidence = choose(
                laya, case[text_lang], SEVERITY_Q[q_lang], SEVERITY_CRITERIA[q_lang])
            rows.append({
                "id": case["id"], "expected": case["severity"], "got": choice,
                "correct": choice == case["severity"], "confidence": round(confidence, 4),
            })
        summary[mode_name(text_lang, q_lang)] = {
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3), "cases": rows}

    record("unresolved_severity", summary)
    assert summary, "sem resultados"


# --- 6. o argumento de eficiencia: custo de varrer um PRD inteiro --------------------------

def test_full_prd_screening_cost(laya, spec_cases):
    """Custo de varrer um PRD inteiro: 13 requisitos, 3 perguntas cada, uma chamada por requisito.

    E este numero que decide se vale acoplar o Laya ao spec-research: se varrer o PRD inteiro
    custa poucos segundos e nenhum token de LLM, a triagem pode rodar sempre, e a LLM so le
    o que foi marcado.
    """
    requirements = spec_cases["requirements"]
    criteria = spec_cases["panel_criteria_en"]
    questions = {
        "ambiguous": {"type": "noul", "instructions": AMBIGUITY_Q["en"]},
        "ui_surface": {"type": "noul", "instructions": UI_SURFACE_Q["en"]},
        "panel": {"type": "choice", "instructions": PANEL_Q["en"], "criteria": criteria},
    }

    laya.call("laya_predict", state=requirements[0]["en"], questions=questions)  # warm

    started = time.monotonic()
    screened = []
    for case in requirements:
        result = laya.call("laya_predict", state=case["en"], questions=questions)
        answers = result["answers"]
        screened.append({
            "id": case["id"],
            "panel": answers["panel"]["choice"],
            "ambiguous": answers["ambiguous"]["noul"] >= 0.3,
            "ui_surface": answers["ui_surface"]["noul"] >= 0.5,
        })
    elapsed = time.monotonic() - started

    record("full_prd_screen", {
        "requirements": len(requirements),
        "questions_per_requirement": len(questions),
        "total_questions": len(requirements) * len(questions),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_requirement": round(elapsed / len(requirements), 3),
        "flagged_ambiguous": sum(r["ambiguous"] for r in screened),
        "flagged_ui": sum(r["ui_surface"] for r in screened),
        "panels_touched": sorted({r["panel"] for r in screened}),
        "screened": screened,
    })

    assert elapsed < 60, f"varrer {len(requirements)} requisitos levou {elapsed:.1f}s"


def test_cost_of_packing_versus_separate_calls(laya, spec_cases):
    """Custo de empacotar perguntas contra chamadas separadas, medido em vez de assumido.

    O material do Laya promete ganho ao empacotar (um forward pass para todas). Em GPU isso
    se sustenta; aqui, em CPU, a sequencia empacotada fica mais longa e o resultado se
    inverte. O teste registra o que medir der e so guarda o custo por pergunta.
    """
    text = spec_cases["requirements"][0]["en"]
    ambiguous = {"ambiguous": {"type": "noul", "instructions": AMBIGUITY_Q["en"]}}
    ui = {"ui_surface": {"type": "noul", "instructions": UI_SURFACE_Q["en"]}}
    panel = {"panel": {"type": "choice", "instructions": PANEL_Q["en"],
                       "criteria": spec_cases["panel_criteria_en"]}}
    packed = {**ambiguous, **ui, **panel}

    laya.call("laya_predict", state=text, questions=packed)  # warm

    def timed(questions, repeats=3):
        samples = []
        for _ in range(repeats):
            started = time.monotonic()
            laya.call("laya_predict", state=text, questions=questions)
            samples.append((time.monotonic() - started) * 1000)
        return statistics.median(samples)

    ambiguous_ms = timed(ambiguous)
    ui_ms = timed(ui)
    panel_ms = timed(panel)
    packed_ms = timed(packed)
    separate_ms = ambiguous_ms + ui_ms + panel_ms

    record("question_packing", {
        "noul_ambiguous_ms": round(ambiguous_ms, 1),
        "noul_ui_ms": round(ui_ms, 1),
        "choice_panel_6_options_ms": round(panel_ms, 1),
        "three_separate_calls_ms": round(separate_ms, 1),
        "three_packed_in_one_call_ms": round(packed_ms, 1),
        "packing_saving": round(1 - packed_ms / separate_ms, 3),
        "choice_vs_noul_ratio": round(panel_ms / ambiguous_ms, 2),
    })

    # Nenhuma direcao e afirmada: o ponto e ter o numero medido nesta maquina. O que
    # precisa valer e que a triagem por requisito continue barata.
    cheapest_per_requirement = min(packed_ms, separate_ms)
    assert cheapest_per_requirement < 5000, (
        f"triar um requisito custa {cheapest_per_requirement:.0f}ms - caro demais para varrer PRD"
    )


# --- 7. a receita recomendada, medida ponta a ponta ----------------------------------------

def test_confidence_gated_panel_routing_is_safe_to_automate(laya, spec_cases):
    """A receita aprovada: PRD em portugues, criterios em ingles, corte de confianca.

    Mede o que interessa para confiar nisso: dos requisitos aceitos sem revisao, quantos
    estavam certos (precisao), e quanto trabalho sobra para a LLM.
    """
    criteria = spec_cases["panel_criteria_en"]
    rows = []
    for case in spec_cases["requirements"]:
        choice, confidence = choose(laya, case["pt"], PANEL_Q["en"], criteria)
        rows.append({"id": case["id"], "expected": case["panel"], "got": choice,
                     "correct": choice == case["panel"], "confidence": confidence})

    gates = {}
    for gate in (0.0, 0.2, 0.3, 0.4, 0.5):
        accepted = [r for r in rows if r["confidence"] >= gate]
        escalated = [r for r in rows if r["confidence"] < gate]
        wrong_accepted = [r for r in accepted if not r["correct"]]
        gates[str(gate)] = {
            "auto_accepted": len(accepted),
            "escalated_to_llm": len(escalated),
            "precision_on_accepted": round(
                sum(r["correct"] for r in accepted) / len(accepted), 3) if accepted else None,
            "wrong_but_accepted": [r["id"] for r in wrong_accepted],
        }

    record("gated_panel_routing", {"gates": gates, "cases": [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]})

    # Precisa existir um corte que entregue roteamento perfeito no que for aceito sozinho,
    # ainda sobrando trabalho automatizado (senao a triagem nao serviu para nada).
    safe = {g: s for g, s in gates.items()
            if s["precision_on_accepted"] == 1.0 and s["auto_accepted"] >= len(rows) // 2}
    record("gated_panel_routing_safe_gates", safe)
    assert safe, f"nenhum corte de confianca automatiza metade do PRD sem errar: {gates}"
