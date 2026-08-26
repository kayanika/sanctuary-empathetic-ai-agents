"""S-10 & S-11 tests — Agent 3 oversight (temporal, fabrication, exclusion).

Unit tests for _parse_verdict need no LLM. The agent tests each make one Ollama
call and hand-craft an error case.
"""
from dotenv import load_dotenv

load_dotenv()

from src.agents.oversight import oversight_agent


def _run(symptoms, hypothesis, phq_tier="moderate") -> dict:
    state = {
        "user_input": "",
        "multimodal_features": {},
        "dialogue_history": [],
        "extracted_symptoms": symptoms,
        "diagnostic_hypothesis": hypothesis,
        "retrieved_passages": [],
        "phq_tier": phq_tier,
        "audit_notes": "",
        "assessment_approved": False,
        "oversight_corrections": [],
        "recommended_intervention": "",
        "crisis_escalation": False,
        "aar_trace": [],
    }
    return oversight_agent(state)


# --- _parse_verdict: deterministic, no LLM ---

def test_parse_verdict_clean_json():
    from src.agents.oversight import _parse_verdict
    v = _parse_verdict('{"temporal_violation": true, "fabrication_violation": false, "exclusion_violation": false, "explanation": "only 5 days", "confidence": 0.9}')
    assert v["temporal_violation"] is True
    assert v["confidence"] == 0.9


def test_parse_verdict_fenced():
    from src.agents.oversight import _parse_verdict
    v = _parse_verdict('```json\n{"temporal_violation": false, "fabrication_violation": false, "exclusion_violation": false, "explanation": "ok", "confidence": 0.8}\n```')
    assert v["temporal_violation"] is False


def test_parse_verdict_all_flags():
    from src.agents.oversight import _parse_verdict
    v = _parse_verdict('{"temporal_violation": false, "fabrication_violation": true, "exclusion_violation": false, "explanation": "cites insomnia not reported", "confidence": 0.85}')
    assert v["fabrication_violation"] is True
    assert v["temporal_violation"] is False
    assert v["exclusion_violation"] is False


def test_parse_verdict_garbage_flags_all_classes():
    """Unparseable output flags EVERY class for review, never approves."""
    from src.agents.oversight import _parse_verdict
    v = _parse_verdict("looks fine to me")
    assert v["temporal_violation"] is True
    assert v["fabrication_violation"] is True
    assert v["exclusion_violation"] is True


# --- oversight agent: one LLM call each ---

def test_catches_temporal_violation():
    """E1: MDD asserted on only 5 days of symptoms must be rejected."""
    result = _run(
        symptoms=["depressed mood 5 days", "fatigue 5 days", "poor concentration 5 days"],
        hypothesis="The patient meets full criteria for a major depressive episode and should be classified as moderate depression.",
    )
    assert result["assessment_approved"] is False
    assert len(result["oversight_corrections"]) > 0


def test_valid_duration_not_flagged():
    """Adequate duration (>3 weeks) should not be flagged as a temporal violation."""
    result = _run(
        symptoms=["depressed mood >3 weeks", "fatigue >3 weeks", "anhedonia >3 weeks"],
        hypothesis="Symptoms have persisted well beyond 2 weeks; the presentation is consistent with a major depressive episode.",
    )
    assert result["assessment_approved"] is True


def test_catches_fabrication():
    """E2: a hypothesis citing symptoms never reported must be flagged (LLM call)."""
    result = _run(
        symptoms=["low mood >3 weeks", "fatigue >3 weeks"],
        hypothesis="The patient reports insomnia, weight loss, and recurrent suicidal ideation, meeting full criteria for severe major depressive disorder.",
    )
    assert result["assessment_approved"] is False
    assert result["aar_trace"][0]["fabrication_violation"] is True
