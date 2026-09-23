"""Decision-quality battery: does Laya actually answer these questions correctly?

Thresholds are regression guards set below measured behaviour, not accuracy targets. The
point is to catch a checkpoint, routing or prompt-shape change that degrades the answers,
while tolerating the model's known weak spots (documented in tests/TESTS.md).

Every metric computed here is written to tests/reports/metrics.json for the write-up.
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
def report():
    yield _metrics
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(_metrics, indent=2, ensure_ascii=False))


def record(name: str, value) -> None:
    _metrics[name] = value


# --- choice: department triage ---------------------------------------------------------

def test_department_triage_accuracy(laya, cases):
    """Department routing over 15 labelled tickets, English and Portuguese."""
    criteria = cases["department_criteria"]
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": criteria,
        }
    }

    rows = []
    for case in cases["triage"]:
        result = laya.call("laya_predict", state=case["text"], questions=questions)
        answer = result["answers"]["department"]
        rows.append(
            {
                "id": case["id"],
                "lang": case["lang"],
                "expected": case["department"],
                "got": answer["choice"],
                "correct": answer["choice"] == case["department"],
                "confidence": answer["confidence"],
                "routed_to": result["routing"]["model"],
            }
        )

    overall = sum(r["correct"] for r in rows) / len(rows)
    by_lang = {}
    for lang in {r["lang"] for r in rows}:
        subset = [r for r in rows if r["lang"] == lang]
        by_lang[lang] = sum(r["correct"] for r in subset) / len(subset)

    record("triage", {"n": len(rows), "accuracy": round(overall, 3),
                      "by_language": {k: round(v, 3) for k, v in by_lang.items()},
                      "cases": rows})

    wrong = [f"{r['id']}: expected {r['expected']}, got {r['got']} ({r['confidence']:.2f})"
             for r in rows if not r["correct"]]
    assert overall >= 0.80, f"department accuracy {overall:.3f} below guard 0.80\n" + "\n".join(wrong)


def test_triage_works_in_portuguese_not_just_english(laya, cases):
    """Guards the router: a Latin-script non-English state must not collapse."""
    criteria = cases["department_criteria"]
    questions = {"department": {"type": "choice",
                                "instructions": "Which department should handle this request?",
                                "criteria": criteria}}
    pt = [c for c in cases["triage"] if c["lang"] == "pt"]
    correct = 0
    for case in pt:
        result = laya.call("laya_predict", state=case["text"], questions=questions)
        correct += result["answers"]["department"]["choice"] == case["department"]

    accuracy = correct / len(pt)
    record("triage_pt_accuracy", round(accuracy, 3))
    assert accuracy >= 0.60, f"Portuguese triage accuracy {accuracy:.3f} below guard 0.60"


# --- noul: boolean judgements ----------------------------------------------------------

def _noul_battery(laya, items, instructions):
    rows = []
    for case in items:
        result = laya.call(
            "laya_predict",
            state=case["text"],
            questions={"q": {"type": "noul", "instructions": instructions}},
        )
        probability = result["answers"]["q"]["noul"]
        rows.append({
            "id": case["id"],
            "expected": case["expected"],
            "probability": probability,
            "correct": (probability >= 0.5) == case["expected"],
        })
    return rows


def _separation(rows) -> float:
    """Mean probability on the positives minus on the negatives. Higher is better."""
    yes = [r["probability"] for r in rows if r["expected"]]
    no = [r["probability"] for r in rows if not r["expected"]]
    return statistics.mean(yes) - statistics.mean(no)


def test_refund_detection(laya, cases):
    """noul: does the user explicitly ask for a refund - 6 labelled cases."""
    rows = _noul_battery(laya, cases["refund_noul"], "Does the user explicitly request a refund?")
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    separation = _separation(rows)
    record("refund_noul", {"n": len(rows), "accuracy": round(accuracy, 3),
                           "separation": round(separation, 3), "cases": rows})

    wrong = [f"{r['id']}: expected {r['expected']}, p={r['probability']:.3f}" for r in rows if not r["correct"]]
    assert accuracy >= 0.80, "refund detection below guard 0.80\n" + "\n".join(wrong)
    assert separation >= 0.30, f"positives and negatives barely separated ({separation:.3f})"


def test_churn_risk_detection(laya, cases):
    """noul: does the user threaten to cancel - 4 labelled cases."""
    rows = _noul_battery(laya, cases["churn_noul"], "Does the user threaten to cancel or leave?")
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    separation = _separation(rows)
    record("churn_noul", {"n": len(rows), "accuracy": round(accuracy, 3),
                          "separation": round(separation, 3), "cases": rows})

    wrong = [f"{r['id']}: expected {r['expected']}, p={r['probability']:.3f}" for r in rows if not r["correct"]]
    assert accuracy >= 0.75, "churn detection below guard 0.75\n" + "\n".join(wrong)


# --- score: ordinal ---------------------------------------------------------------------

def test_urgency_score_orders_blocking_above_trivial(laya, cases):
    """`score` is the model's weakest primitive, so this only asserts the ordering."""
    questions = {"urgency": {"type": "score",
                             "instructions": "How urgent is this request?",
                             "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]}}
    scores = {}
    for case in cases["urgency_score"]:
        result = laya.call("laya_predict", state=case["text"], questions=questions)
        scores[case["id"]] = result["answers"]["urgency"]["score"]

    record("urgency_scores", {k: round(v, 3) for k, v in scores.items()})
    assert scores["urg-high"] > scores["urg-low"], f"urgency not ordered: {scores}"


# --- routing ----------------------------------------------------------------------------

def test_router_sends_non_latin_scripts_to_the_multilingual_checkpoint(laya, cases):
    """The English checkpoint stays confident while being wrong on these scripts."""
    rows = []
    for case in cases["routing"]:
        result = laya.call(
            "laya_predict",
            state=case["text"],
            questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
        )
        rows.append({
            "id": case["id"],
            "expected_model": case["expected_model"],
            "routed_to": result["routing"]["model"],
            "reason": result["routing"]["reason"],
            "refund_probability": result["answers"]["refund"]["noul"],
        })

    record("routing", rows)
    wrong = [r for r in rows if r["routed_to"] != r["expected_model"]]
    assert not wrong, f"router picked the wrong checkpoint: {wrong}"


def test_non_latin_refund_intent_survives_routing(laya, cases):
    """End-to-end proof the routing is worth something: same intent, four scripts."""
    rows = []
    for case in cases["routing"]:
        result = laya.call(
            "laya_predict",
            state=case["text"],
            questions={"refund": {"type": "noul", "instructions": "Is the user asking for a refund?"}},
        )
        rows.append({"id": case["id"], "probability": result["answers"]["refund"]["noul"]})

    record("multilingual_refund_intent", rows)
    detected = sum(r["probability"] >= 0.5 for r in rows)
    assert detected >= len(rows) - 1, f"refund intent lost in translation: {rows}"


# --- behaviour under composition ---------------------------------------------------------

def test_answer_is_stable_whether_asked_alone_or_alongside_others(laya, cases):
    """Extra questions share one forward pass; they must not change each other's answers."""
    state = "The checkout page returns HTTP 500 on every order since the deploy."
    criteria = cases["department_criteria"]
    department = {"type": "choice",
                  "instructions": "Which department should handle this request?",
                  "criteria": criteria}

    alone = laya.call("laya_predict", state=state, questions={"department": department})
    together = laya.call("laya_predict", state=state, questions={
        "department": department,
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "soon", "blocking"]},
        "churn_risk": {"type": "noul", "instructions": "Does the user threaten to leave?"},
        "refund_requested": {"type": "noul", "instructions": "Is a refund requested?"},
    })

    a = alone["answers"]["department"]
    b = together["answers"]["department"]
    record("composition_stability", {"alone": a["choice"], "alone_confidence": a["confidence"],
                                     "together": b["choice"], "together_confidence": b["confidence"]})
    assert a["choice"] == b["choice"], "answer changed when other questions were added"


def test_repeated_calls_are_deterministic(laya):
    """No sampling: the same state and questions must give byte-identical probabilities."""
    payload = dict(
        state="Billed twice again. Refund us today or we cancel the contract.",
        questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    )
    first = laya.call("laya_predict", **payload)["answers"]["refund"]["noul"]
    second = laya.call("laya_predict", **payload)["answers"]["refund"]["noul"]
    record("determinism", {"first": first, "second": second})
    assert first == second, f"non-deterministic output: {first} != {second}"


def test_classify_matches_the_equivalent_predict_call(laya, cases):
    """laya_classify is a shortcut, not a different model path."""
    text = "We would like a quote for 200 enterprise seats."
    labels = cases["department_criteria"]
    instructions = "Which department should handle this request?"

    shortcut = laya.call("laya_classify", text=text, labels=labels, instructions=instructions)
    direct = laya.call("laya_predict", state=text, questions={
        "label": {"type": "choice", "instructions": instructions, "criteria": labels}})

    assert shortcut["label"] == direct["answers"]["label"]["choice"]
    assert shortcut["confidence"] == direct["answers"]["label"]["confidence"]


# --- documented limits: measured, not asserted away ---------------------------------------

def test_high_cardinality_choice_degrades_as_documented(laya):
    """Options share a fixed token budget, so many labels blur together.

    This does not assert that the model is good with 40 labels - it asserts the small-label
    path still works, and records the gap so the skill's guidance stays honest.
    """
    text = "The checkout page returns HTTP 500 on every order since the deploy."
    few = {"billing": "payments and invoices",
           "technical": "bugs, outages and errors",
           "sales": "pricing and contracts",
           "other": "anything else"}
    many = dict(few)
    for i in range(36):
        many[f"topic_{i:02d}"] = f"unrelated internal topic number {i}"

    small = laya.call("laya_classify", text=text, labels=few)
    large = laya.call("laya_classify", text=text, labels=many)

    record("high_cardinality", {
        "labels_small": len(few), "label_small": small["label"], "confidence_small": small["confidence"],
        "labels_large": len(many), "label_large": large["label"], "confidence_large": large["confidence"],
        "still_correct_at_scale": large["label"] == "technical",
    })
    assert small["label"] == "technical", "the documented-good path (few labels) regressed"


def test_confidence_is_higher_on_clear_cases_than_ambiguous_ones(laya, cases):
    """Calibration sanity: the model ships over-confident, but ranking should still hold."""
    labels = cases["department_criteria"]
    clear = laya.call("laya_classify", text="Refund the duplicate charge on invoice #4411.", labels=labels)
    vague = laya.call("laya_classify", text="Hi, I have a question about my account.", labels=labels)

    record("confidence_ranking", {"clear": clear["confidence"], "vague": vague["confidence"],
                                  "clear_label": clear["label"], "vague_label": vague["label"]})
    assert clear["confidence"] > vague["confidence"], (
        f"clear case ({clear['confidence']:.3f}) not more confident than vague one ({vague['confidence']:.3f})"
    )


# --- latency -------------------------------------------------------------------------------

def test_latency_stays_in_the_expected_envelope(laya):
    """Round-trip through MCP + HTTP + CPU inference. Not the paper's GPU numbers."""
    one = {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}
    ten = {f"q{i}": {"type": "noul", "instructions": f"Is topic {i} mentioned?"} for i in range(10)}
    state = "Billed twice again. Refund us today or we cancel the contract."

    laya.call("laya_predict", state=state, questions=one)  # warm

    single = []
    for _ in range(5):
        started = time.monotonic()
        laya.call("laya_predict", state=state, questions=one)
        single.append((time.monotonic() - started) * 1000)

    started = time.monotonic()
    laya.call("laya_predict", state=state, questions=ten)
    batched = (time.monotonic() - started) * 1000

    record("latency_ms", {
        "single_question_p50": round(statistics.median(single), 1),
        "single_question_min": round(min(single), 1),
        "single_question_max": round(max(single), 1),
        "ten_questions_batched": round(batched, 1),
        "ten_questions_per_question": round(batched / 10, 1),
    })
    assert statistics.median(single) < 5000, "single-question latency exceeded 5s"
    assert batched < 10000, "ten batched questions exceeded 10s"
