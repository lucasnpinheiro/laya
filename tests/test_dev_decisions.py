"""Laya como apoio a decisoes do dia a dia de desenvolvimento.

Perguntas fechadas que a LLM repete item a item num fluxo de desenvolvimento: que tipo de
commit e este? a issue e bug ou feature? por que o teste falhou? este log acorda o
plantao? a review bloqueia? a mudanca quebra clientes, mexe no schema, toca seguranca?

Cada pergunta e medida em dois modos, sempre com criterios em ingles (o modo vencedor de
SPEC-RESEARCH.md):

  pt/en  texto em portugues, criterios em ingles   (o texto como veio)
  en/en  texto e criterios em ingles               (a LLM traduz antes)

O veredito de cada pergunta (APROVADO / COM PORTAO / REPROVADO) vive em
question_banks/dev_decisions.json e em tests/DEV-DECISIONS.md. Esta bateria so segura o
que foi aprovado: se uma pergunta aprovada cair abaixo do piso, o banco mente.
Metricas vao para tests/reports/dev_decisions_metrics.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

REPORT_DIR = Path(__file__).parent / "reports"
FIXTURE = Path(__file__).parent / "fixtures" / "dev_decisions_cases.json"
BANK = Path(__file__).resolve().parent.parent / "question_banks" / "dev_decisions.json"

NOUL_CUT = 0.5
MODES = ("pt", "en")

_metrics: dict = {}


@pytest.fixture(scope="session", autouse=True)
def dev_report():
    yield _metrics
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "dev_decisions_metrics.json").write_text(
        json.dumps(_metrics, indent=2, ensure_ascii=False)
    )


@pytest.fixture(scope="session")
def dev_cases():
    return json.loads(FIXTURE.read_text())["questions"]


def _question(spec: dict) -> dict:
    q = {"type": spec["type"], "instructions": spec["instructions"]}
    if "criteria" in spec:
        q["criteria"] = spec["criteria"]
    return q


def _measure(laya, name: str, spec: dict, mode: str) -> dict:
    question = {name: _question(spec)}
    rows = []
    for case in spec["cases"]:
        answer = laya.call("laya_predict", state=case[mode], questions=question)["answers"][name]
        if spec["type"] == "noul":
            predicted = answer["noul"] >= NOUL_CUT
            rows.append({"ok": predicted == case["label"], "value": answer["noul"], "label": case["label"]})
        else:
            rows.append({"ok": answer["choice"] == case["label"], "value": answer["confidence"],
                         "label": case["label"], "got": answer["choice"]})

    right = [r["value"] for r in rows if r["ok"]]
    wrong = [r["value"] for r in rows if not r["ok"]]
    out = {
        "accuracy": round(len(right) / len(rows), 3),
        "n": len(rows),
        "misses": [r for r in rows if not r["ok"]],
        "rows": rows,
    }
    if spec["type"] == "noul":
        yes = [r["value"] for r in rows if r["label"]]
        no = [r["value"] for r in rows if not r["label"]]
        out["mean_p_when_yes"] = round(statistics.mean(yes), 3)
        out["mean_p_when_no"] = round(statistics.mean(no), 3)
        out["separation"] = round(out["mean_p_when_yes"] - out["mean_p_when_no"], 3)
    else:
        out["conf_when_right"] = round(statistics.mean(right), 3) if right else None
        out["conf_when_wrong"] = round(statistics.mean(wrong), 3) if wrong else None
    return out


def test_measure_every_dev_question(laya, dev_cases):
    for name, spec in dev_cases.items():
        _metrics[name] = {mode: _measure(laya, name, spec, mode) for mode in MODES}


def test_bank_verdicts_still_hold(dev_cases):
    """Cada pergunta aprovada no banco tem de continuar no piso que a aprovou."""
    if not _metrics:
        pytest.skip("depends on test_measure_every_dev_question")
    bank = json.loads(BANK.read_text())
    for name, entry in bank["approved"].items():
        mode = entry["_mode"]
        measured = _metrics[name][mode]
        if "_noul_cut" in entry:
            # Sinalizador: todo "sim" acima do corte, todo "nao" abaixo.
            cut = entry["_noul_cut"]
            wrong = [r for r in measured["rows"] if (r["value"] >= cut) != r["label"]]
            assert not wrong, f"{name} [{mode}] nao separa mais no corte {cut}: {wrong}"
        else:
            floor = entry["_accuracy_floor"]
            assert measured["accuracy"] >= floor, (
                f"{name} [{mode}] caiu para {measured['accuracy']}, piso {floor}"
            )
            gate = entry["_confidence_gate"]
            accepted = [r for r in measured["rows"] if r["value"] >= gate]
            assert all(r["ok"] for r in accepted), f"{name} [{mode}] errou acima do portao {gate}"
