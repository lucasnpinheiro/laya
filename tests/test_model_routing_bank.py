"""Banco `model_routing`: a pergunta `effort_level` que o roteador de modelos do ~/.agents usa.

A medida vive em ~/.agents/docs/model-routing-calibration.md (70 briefs sinteticos, 2026-10-09,
checkpoint `english`, chamada combinada `task_tier` + `effort_level`). O veredito e estreito:
Laya so pode BAIXAR um candidato `medium` para `low` quando p(low) >= 0.60, e so para Sonnet
(`padrao`) e Opus-medium. Ela nunca sobe effort: `high` saiu 0/10 e `xhigh` nunca foi medido.

Esta bateria segura o contrato do banco (formato, portao, usos proibidos) e confere que a
pergunta banqueada roda no laya_predict e devolve as tres probabilidades do score. Ela nao
re-mede a acuracia: para isso, re-rode a sonda de 70 briefs descrita no doc acima.
"""

from __future__ import annotations

import pytest

import json
from pathlib import Path

BANK = Path(__file__).resolve().parent.parent / "question_banks" / "model_routing.json"

MECHANICAL_BRIEF = {
    "role": "Explorer",
    "brief": "List every controller file under src/Controller and return the file:line of each public action.",
}


def _bank() -> dict:
    return json.loads(BANK.read_text())


def test_bank_file_declares_measurement_and_source():
    bank = _bank()
    assert bank["_measured_on"] == "2026-10-09"
    assert "model-routing-calibration.md" in bank["_about"]


def test_effort_level_is_approved_for_downgrade_only():
    entry = _bank()["approved"]["effort_level"]
    assert entry["_use"] == "downgrade_only"
    assert entry["_downgrade"] == {"from": "medium", "to": "low"}
    assert entry["_p_low_gate"] == 0.60
    assert sorted(entry["_applies_to"]) == ["opus-medium", "padrao"]
    assert "rapido" in entry["_never_for"]
    q = entry["question"]["effort_level"]
    assert q["type"] == "score"
    assert [c.split(":")[0] for c in q["criteria"]] == ["low", "medium", "high"]


def test_raising_effort_is_rejected():
    rejected = _bank()["rejected"]
    assert "effort_raise_to_high" in rejected
    assert "0/10" in rejected["effort_raise_to_high"]["_verdict"]
    assert "effort_xhigh" in rejected
    assert "never measured" in rejected["effort_xhigh"]["_verdict"]


def test_bank_is_served_by_the_mcp_tool(laya):
    listing = laya.call("laya_question_bank")
    assert "model_routing" in {b["name"] for b in listing["banks"]}
    served = laya.call("laya_question_bank", name="model_routing")
    assert served["approved"]["effort_level"]["_p_low_gate"] == 0.60


@pytest.mark.model
def test_banked_question_returns_three_level_probabilities(laya):
    question = _bank()["approved"]["effort_level"]["question"]
    result = laya.call("laya_predict", state=MECHANICAL_BRIEF, questions=question)
    answer = result["answers"]["effort_level"]
    assert answer["type"] == "score"
    assert set(answer["probabilities"]) == {"0", "1", "2"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 0.01
