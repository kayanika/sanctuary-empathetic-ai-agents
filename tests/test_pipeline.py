"""S-12 tests — approval/rejection gate + crisis short-circuit.

Nearly all tests here need NO LLM: crisis short-circuits before any model call,
and the routing functions are pure.
"""
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END
from src.pipeline.graph import tao_graph, route_after_oversight, route_after_crisis
from src.safety.crisis import check_crisis, crisis_gate
from src.agents.inconclusive import inconclusive_agent


def _base_state(**overrides):
    state = {
        "user_input": "",
        "multimodal_features": {},
        "dialogue_history": [],
        "extracted_symptoms": [],
        "diagnostic_hypothesis": "",
        "retrieved_passages": [],
        "phq_tier": "",
        "audit_notes": "",
        "assessment_approved": False,
        "oversight_corrections": [],
        "oversight_confidence": 0.0,
        "recommended_intervention": "",
        "crisis_escalation": False,
        "aar_trace": [],
    }
    state.update(overrides)
    return state


# --- crisis detection (S-14 criteria: >=10 high-risk, >=5 normal) ---

def test_check_crisis_high_risk_phrasings():
    high_risk = [
        "I want to kill myself",
        "I've been thinking about killing myself",
        "I am suicidal",
        "I want to end my life",
        "thinking about self-harm",
        "I want to die",
        "I don't want to live anymore",
        "there's no reason to live",
        "I keep thinking about hurting myself",
        "I attempted suicide last year",
    ]
    for phrase in high_risk:
        assert check_crisis(phrase) is True, f"missed: {phrase}"


def test_check_crisis_no_false_positives():
    normal = [
        "I've been feeling tired lately",
        "work has been stressful",
        "I can't sleep well",
        "I feel unmotivated most days",
        "my concentration is poor",
    ]
    for phrase in normal:
        assert check_crisis(phrase) is False, f"false positive: {phrase}"


# --- routing decisions: pure functions ---

def test_route_after_oversight_approved():
    assert route_after_oversight(_base_state(assessment_approved=True)) == "routing"


def test_route_after_oversight_rejected_goes_to_inconclusive():
    """A rejected assessment must NOT silently end — it gets a safe output."""
    assert route_after_oversight(_base_state(assessment_approved=False)) == "inconclusive"


def test_route_after_crisis_short_circuits():
    assert route_after_crisis(_base_state(crisis_escalation=True)) == END
    assert route_after_crisis(_base_state(crisis_escalation=False)) == "intake"


# --- terminal nodes ---

def test_inconclusive_gives_safe_output():
    result = inconclusive_agent(_base_state(
        oversight_corrections=["Temporal threshold (E1): only 5 days reported"]
    ))
    assert len(result["recommended_intervention"]) > 0
    assert result["aar_trace"][0]["agent"] == "inconclusive"


def test_crisis_gate_sets_escalation_and_message():
    result = crisis_gate(_base_state(user_input="I want to kill myself"))
    assert result["crisis_escalation"] is True
    assert "988" in result["recommended_intervention"]


# --- full graph: the crisis path needs NO LLM ---

def test_graph_crisis_short_circuits_before_any_llm():
    """The regression test for the bug this story fixed: crisis must escalate
    without any LLM agent running, even though oversight would have rejected."""
    result = tao_graph.invoke(_base_state(
        user_input="I've only felt this way 5 days but I keep thinking about killing myself"
    ))
    assert result["crisis_escalation"] is True
    assert "988" in result["recommended_intervention"]
    agents = [e["agent"] for e in result["aar_trace"]]
    assert agents == ["crisis_gate"], f"LLM agents must not run on crisis: {agents}"
