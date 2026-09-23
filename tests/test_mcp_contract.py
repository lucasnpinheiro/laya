"""Contract tests: the MCP surface itself — tools, schemas, validation, error propagation.

These do not judge how good the model is (see test_decision_quality.py); they check that
a client can discover the tools, that bad input is rejected before it reaches the API, and
that failure messages survive the trip back to the client.
"""

from __future__ import annotations

EXPECTED_TOOLS = {
    "laya_predict",
    "laya_classify",
    "laya_health",
    "laya_start",
    "laya_checkpoints",
}


def test_server_exposes_the_documented_tools(laya):
    """All five documented tools show up in client discovery."""
    names = {tool.name for tool in laya.list_tools()}
    assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"


def test_every_tool_is_described(laya):
    """Every tool carries a description a client can choose from."""
    for tool in laya.list_tools():
        assert tool.description and len(tool.description) > 40, f"{tool.name} lacks a usable description"


def test_predict_schema_advertises_its_arguments(laya):
    """The laya_predict schema declares state, questions and model, the first two required."""
    predict = next(t for t in laya.list_tools() if t.name == "laya_predict")
    props = predict.input_schema["properties"]
    assert {"state", "questions", "model"} <= set(props)
    assert set(predict.input_schema.get("required", [])) == {"state", "questions"}


def test_health_reports_a_loaded_model(laya):
    """laya_health reports the API up, the model loaded and auto-start available."""
    health = laya.call("laya_health")
    assert health["reachable"] is True
    assert health["model_loaded"] is True
    assert health["auto_start"]["enabled"] is True
    assert health["auto_start"]["possible"] is True


def test_checkpoints_are_listed_with_their_budgets(laya):
    """laya_checkpoints lists the three checkpoints with each one's context budget and warning."""
    checkpoints = laya.call("laya_checkpoints")
    by_name = {c["name"]: c for c in checkpoints}
    assert set(by_name) == {"english", "multilingual", "typed-decisions"}
    assert by_name["english"]["context"] == 512
    assert by_name["multilingual"]["context"] == 1024
    assert all(c["warning"] for c in checkpoints)


# --- input validation: rejected locally, never forwarded to the API -------------------

def test_choice_without_criteria_is_rejected(laya):
    """A choice question without criteria is refused before it reaches the API."""
    message = laya.call_expecting_error(
        "laya_predict",
        state="anything",
        questions={"dept": {"type": "choice", "instructions": "which one?"}},
    )
    assert "criteria" in message and "dept" in message


def test_choice_with_a_single_option_is_rejected(laya):
    """A choice question with a single option is refused."""
    message = laya.call_expecting_error(
        "laya_predict",
        state="anything",
        questions={"dept": {"type": "choice", "instructions": "which?", "criteria": {"only": "one"}}},
    )
    assert "two" in message


def test_score_criteria_must_be_an_ordered_list(laya):
    """A score question needs criteria as an ordered list, not an object."""
    message = laya.call_expecting_error(
        "laya_predict",
        state="anything",
        questions={"urgency": {"type": "score", "instructions": "how urgent?", "criteria": {"low": "x"}}},
    )
    assert "list" in message


def test_noul_must_not_carry_criteria(laya):
    """A noul question carrying criteria is refused - the type takes no options."""
    message = laya.call_expecting_error(
        "laya_predict",
        state="anything",
        questions={"refund": {"type": "noul", "instructions": "refund?", "criteria": ["yes", "no"]}},
    )
    assert "noul" in message


def test_unknown_question_type_is_rejected(laya):
    """An unknown question type is refused, naming the valid ones."""
    message = laya.call_expecting_error(
        "laya_predict",
        state="anything",
        questions={"q": {"type": "regression", "instructions": "predict the price"}},
    )
    assert "choice" in message and "noul" in message


def test_missing_instructions_is_rejected(laya):
    """A question without instructions is refused."""
    message = laya.call_expecting_error(
        "laya_predict",
        state="anything",
        questions={"q": {"type": "noul"}},
    )
    assert "instructions" in message


def test_empty_questions_is_rejected(laya):
    """An empty questions object is refused."""
    message = laya.call_expecting_error("laya_predict", state="anything", questions={})
    assert "non-empty" in message


def test_classify_needs_at_least_two_labels(laya):
    """laya_classify with a single label is refused."""
    message = laya.call_expecting_error("laya_classify", text="anything", labels={"only": "one"})
    assert "two" in message


def test_error_messages_are_specific_not_generic(laya):
    """Regression guard: mcp 2.x only forwards ToolError text, everything else is flattened."""
    message = laya.call_expecting_error("laya_predict", state="x", questions={})
    assert message.strip() != "Error executing tool laya_predict"
    assert len(message) > 40


# --- shape of a successful call --------------------------------------------------------

def test_predict_returns_answers_and_routing_metadata(laya):
    """The result carries answers plus routing with model, repo and the reason for the pick."""
    result = laya.call(
        "laya_predict",
        state="Please refund the duplicate charge.",
        questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    )
    assert "answers" in result and "routing" in result
    assert set(result["routing"]) >= {"model", "repo", "reason"}
    answer = result["answers"]["refund"]
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0


def test_choice_answer_carries_a_full_probability_distribution(laya):
    """A choice answer carries the full distribution summing to 1.0, not just the winner."""
    result = laya.call(
        "laya_classify",
        text="The checkout page returns HTTP 500 on every order.",
        labels={"billing": "payments", "technical": "bugs and outages", "sales": "pricing"},
    )
    probs = result["raw"]["answers"]["label"]["probabilities"]
    assert set(probs) == {"billing", "technical", "sales"}
    assert abs(sum(probs.values()) - 1.0) < 0.02, f"probabilities do not sum to 1: {probs}"
    assert 0.0 <= result["confidence"] <= 1.0


def test_a_state_object_is_accepted_alongside_a_plain_string(laya):
    """state accepts an object of named fields, not only a plain string."""
    result = laya.call(
        "laya_predict",
        state={
            "from": "user@acme.com",
            "subject": "Duplicate charge on invoice #4411",
            "body": "We were billed twice for March. Please refund the duplicate.",
        },
        questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    )
    assert result["answers"]["refund"]["noul"] > 0.5


def test_all_questions_are_answered_in_one_call(laya):
    """The whole point of the model: N questions, one forward pass."""
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing", "other": "rest"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        "churn_risk": {"type": "noul", "instructions": "Does the user threaten to leave?"},
        "refund_requested": {"type": "noul", "instructions": "Is a refund requested?"},
    }
    result = laya.call(
        "laya_predict",
        state="Billed twice again. Refund us today or we cancel the contract.",
        questions=questions,
    )
    assert set(result["answers"]) == set(questions)


def test_explicit_checkpoint_override_is_honoured(laya):
    """Forcing model overrides the router's pick."""
    result = laya.call(
        "laya_predict",
        state="Please refund the duplicate charge.",
        questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
        model="multilingual",
    )
    assert result["routing"]["model"] == "multilingual"


def test_unknown_checkpoint_name_is_refused(laya):
    """An unknown checkpoint name is refused."""
    message = laya.call_expecting_error(
        "laya_predict",
        state="anything",
        questions={"refund": {"type": "noul", "instructions": "refund?"}},
        model="does-not-exist",
    )
    assert message


def test_start_is_idempotent_when_already_running(laya):
    """laya_start on a running container reports already-running and restarts nothing."""
    result = laya.call("laya_start")
    assert result["ready"] is True
    assert result["already_running"] is True
    assert result["elapsed_seconds"] < 30
