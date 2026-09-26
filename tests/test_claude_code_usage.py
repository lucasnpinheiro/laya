"""Laya sobre o historico real do Claude Code.

Perguntas fechadas que o agente repete a cada prompt do desenvolvedor: que tipo de pedido e
este? que area do sistema ele toca? o prompt traz uma senha ou token? pede para apagar
dados? e so uma continuacao da conversa anterior?

Os casos vem de ~/.claude/history.jsonl, parafraseados e anonimizados (nenhum segredo real
na fixture). Cada pergunta e medida em tres modos, sempre com criterios em ingles:

  pt     texto em portugues, checkpoint escolhido pelo router  (o prompt como veio)
  pt_ml  texto em portugues, model="multilingual" forcado      (PT curto cai no english)
  en     texto traduzido para ingles, router                   (a LLM traduz antes)

Para cada modo o relatorio acha o portao de confianca (choice) ou o corte (noul) que
deixaria a pergunta segura, e quanto ela cobriria. O veredito vive em
question_banks/claude_code_usage.json e tests/CLAUDE-CODE-USAGE.md; esta bateria segura
o que foi aprovado. Metricas em tests/reports/claude_code_usage_metrics.json.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

REPORT_DIR = Path(__file__).parent / "reports"
FIXTURE = Path(__file__).parent / "fixtures" / "claude_code_usage_cases.json"
BANK = Path(__file__).resolve().parent.parent / "question_banks" / "claude_code_usage.json"

MODES = {
    "pt": {"text": "pt", "model": None},
    "pt_ml": {"text": "pt", "model": "multilingual"},
    "en": {"text": "en", "model": None},
}

_metrics: dict = {}


@pytest.fixture(scope="session", autouse=True)
def usage_report():
    yield _metrics
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "claude_code_usage_metrics.json").write_text(
        json.dumps(_metrics, indent=2, ensure_ascii=False)
    )


@pytest.fixture(scope="session")
def usage_cases():
    return json.loads(FIXTURE.read_text())["questions"]


def _question(spec: dict) -> dict:
    q = {"type": spec["type"], "instructions": spec["instructions"]}
    if "criteria" in spec:
        q["criteria"] = spec["criteria"]
    return q


def _best_gate(rows: list[dict]) -> dict:
    """Menor portao de confianca em que tudo aceito esta certo (precisao 1.0)."""
    for gate in sorted({r["value"] for r in rows}):
        accepted = [r for r in rows if r["value"] >= gate]
        if accepted and all(r["ok"] for r in accepted):
            return {"gate": math.floor(gate * 1000) / 1000, "coverage": round(len(accepted) / len(rows), 3)}
    return {"gate": None, "coverage": 0.0}


def _best_cut(rows: list[dict]) -> dict:
    """Corte de noul que mais acerta, e se algum corte separa sim de nao por inteiro."""
    best = {"best_cut": 0.5, "best_cut_accuracy": 0.0}
    for cut in sorted({r["value"] for r in rows} | {0.5}):
        acc = sum((r["value"] >= cut) == r["label"] for r in rows) / len(rows)
        if acc > best["best_cut_accuracy"]:
            best = {"best_cut": round(cut, 3), "best_cut_accuracy": round(acc, 3)}
    yes = [r["value"] for r in rows if r["label"]]
    no = [r["value"] for r in rows if not r["label"]]
    best["clean_split"] = min(yes) > max(no)
    # Recall de sinalizador: maior corte que ainda pega todo "sim", e quantos "nao" levanta junto.
    flag_cut = min(yes)
    best["flag_cut_all_yes"] = round(flag_cut, 3)
    best["false_flags_at_that_cut"] = sum(v >= flag_cut for v in no)
    return best


def _measure(laya, name: str, spec: dict, mode: str) -> dict:
    cfg = MODES[mode]
    question = {name: _question(spec)}
    rows = []
    for case in spec["cases"]:
        kwargs = {"state": case[cfg["text"]], "questions": question}
        if cfg["model"]:
            kwargs["model"] = cfg["model"]
        result = laya.call("laya_predict", **kwargs)
        answer = result["answers"][name]
        routed = result.get("routing", {}).get("model")
        if spec["type"] == "noul":
            rows.append({"ok": (answer["noul"] >= 0.5) == case["label"], "value": answer["noul"],
                         "label": case["label"], "routed": routed})
        else:
            rows.append({"ok": answer["choice"] == case["label"], "value": answer["confidence"],
                         "label": case["label"], "got": answer["choice"], "routed": routed})

    right = [r["value"] for r in rows if r["ok"]]
    wrong = [r["value"] for r in rows if not r["ok"]]
    out = {
        "accuracy": round(len(right) / len(rows), 3),
        "n": len(rows),
        "routed": sorted({r["routed"] for r in rows if r["routed"]}),
    }
    if spec["type"] == "noul":
        yes = [r["value"] for r in rows if r["label"]]
        no = [r["value"] for r in rows if not r["label"]]
        out["mean_p_when_yes"] = round(statistics.mean(yes), 3)
        out["mean_p_when_no"] = round(statistics.mean(no), 3)
        out["separation"] = round(out["mean_p_when_yes"] - out["mean_p_when_no"], 3)
        out.update(_best_cut(rows))
    else:
        out["conf_when_right"] = round(statistics.mean(right), 3) if right else None
        out["conf_when_wrong"] = round(statistics.mean(wrong), 3) if wrong else None
        out.update(_best_gate(rows))
        out["confusions"] = sorted(f"{r['label']}->{r['got']}" for r in rows if not r["ok"])
    out["rows"] = rows
    return out


def test_measure_every_usage_question(laya, usage_cases):
    for name, spec in usage_cases.items():
        _metrics[name] = {mode: _measure(laya, name, spec, mode) for mode in MODES}


def test_bank_verdicts_still_hold(usage_cases):
    """Cada pergunta aprovada no banco tem de continuar no piso que a aprovou."""
    if not _metrics:
        pytest.skip("depends on test_measure_every_usage_question")
    if not BANK.exists():
        pytest.skip("bank not written yet - first run only measures")
    bank = json.loads(BANK.read_text())
    for name, entry in bank["approved"].items():
        rows = _metrics[name][entry["_mode"]]["rows"]
        if "_noul_cut" in entry:
            cut = entry["_noul_cut"]
            missed = [r for r in rows if r["label"] and r["value"] < cut]
            assert not missed, f"{name} [{entry['_mode']}] deixou passar 'sim' abaixo do corte {cut}: {missed}"
            false_flags = sum(1 for r in rows if not r["label"] and r["value"] >= cut)
            assert false_flags <= entry.get("_max_false_flags", 0), (
                f"{name} [{entry['_mode']}] levantou {false_flags} falsos alarmes no corte {cut}"
            )
        else:
            gate = entry["_confidence_gate"]
            accepted = [r for r in rows if r["value"] >= gate]
            assert all(r["ok"] for r in accepted), f"{name} [{entry['_mode']}] errou acima do portao {gate}"
            coverage = len(accepted) / len(rows)
            assert coverage >= entry["_min_coverage"], (
                f"{name} [{entry['_mode']}] cobertura {coverage:.2f} < {entry['_min_coverage']}"
            )
