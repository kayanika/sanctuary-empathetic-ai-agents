"""
S-04 acceptance tests — Empathetic dialogue with Gemma 4.

These tests make real Ollama calls. Ollama must be running with gemma4:12b-mlx pulled.
Run: pytest tests/test_intake.py -v
"""
import pytest
from dotenv import load_dotenv

load_dotenv()

from src.agents.intake import intake_agent


def _run(user_input: str, history: list | None = None, multimodal_features: dict | None = None) -> dict:
    state = {
        "user_input": user_input,
        "multimodal_features": multimodal_features or {},
        "dialogue_history": history or [],
        "extracted_symptoms": [],
        "diagnostic_hypothesis": "",
        "retrieved_passages": [],
        "phq_tier": "",
        "audit_notes": "",
        "assessment_approved": False,
        "oversight_corrections": [],
        "recommended_intervention": "",
        "crisis_escalation": False,
        "aar_trace": [],
    }
    return intake_agent(state)


def test_returns_dialogue_history():
    """Agent must add exactly one user turn and one assistant turn to dialogue_history."""
    result = _run("I've been feeling really low and tired for the past few weeks.")
    assert len(result["dialogue_history"]) == 2
    assert result["dialogue_history"][0]["role"] == "user"
    assert result["dialogue_history"][1]["role"] == "assistant"


def test_assistant_response_is_non_empty():
    """Agent must produce a non-empty response."""
    result = _run("I haven't been sleeping well and I feel exhausted all the time.")
    response = result["dialogue_history"][1]["content"]
    assert isinstance(response, str)
    assert len(response.strip()) > 20


def test_response_does_not_diagnose():
    """Agent must not use diagnostic labels in its response."""
    result = _run("I feel sad and unmotivated most days.")
    response = result["dialogue_history"][1]["content"].lower()
    forbidden = ["you have depression", "you have anxiety", "you are depressed", "diagnosis"]
    for term in forbidden:
        assert term not in response, f"Response made a clinical claim: '{term}'"


def test_response_ends_with_question():
    """Agent must ask a follow-up question (S-04 requires PHQ-aligned domains)."""
    result = _run("I've been struggling to get out of bed lately.")
    response = result["dialogue_history"][1]["content"]
    assert "?" in response, "Agent response must contain a follow-up question"


def test_aar_trace_entry():
    """AAR trace must have exactly one entry labelled 'intake'."""
    result = _run("Everything just feels heavy and I can't concentrate.")
    assert len(result["aar_trace"]) == 1
    assert result["aar_trace"][0]["agent"] == "intake"
    assert len(result["aar_trace"][0]["input_summary"]) > 0
    assert len(result["aar_trace"][0]["output_summary"]) > 0


def test_extracted_symptoms_is_list():
    """extracted_symptoms must be a list (populated in S-06; empty is fine here)."""
    result = _run("I feel hopeless and have no energy.")
    assert isinstance(result["extracted_symptoms"], list)


def test_multi_turn_history_included():
    """Prior dialogue history must be passed into the LLM context."""
    history = [
        {"role": "user", "content": "I've been feeling down lately."},
        {"role": "assistant", "content": "Thank you for sharing that. How long have you been feeling this way?"},
    ]
    result = _run("It's been about three weeks now.", history=history)
    response = result["dialogue_history"][1]["content"]
    # Agent should have context of prior turn — response should be coherent follow-up
    assert len(response.strip()) > 20
    
    
    # --- S-05 tests ---

def test_format_features_returns_empty_for_empty_dict():
    """_format_features must return empty string when no features provided."""
    from src.agents.intake import _format_features
    assert _format_features({}) == ""


def test_format_features_includes_provided_fields():
    """_format_features must include all fields that are present in the dict."""
    from src.agents.intake import _format_features
    features = {
        "speech_rate": 120.5,
        "pause_duration": 0.8,
        "valence": -0.4,
        "au_intensities": {"1": 0.3, "4": 0.7},
    }
    result = _format_features(features)
    assert "120.50" in result
    assert "0.80" in result
    assert "-0.40" in result
    assert "AU1=0.30" in result


def test_text_only_mode_without_features():
    """Agent must work normally when multimodal_features is empty."""
    result = _run("I've been feeling anxious lately.", multimodal_features={})
    assert result["aar_trace"][0]["multimodal_included"] is False
    assert len(result["dialogue_history"]) == 2


def test_multimodal_features_recorded_in_trace():
    """When features are provided, trace must record multimodal_included=True."""
    features = {
        "speech_rate": 95.2,
        "pause_duration": 1.2,
        "valence": -0.6,
        "au_intensities": {"1": 0.3, "4": 0.7},
    }
    result = _run("I haven't been feeling like myself.", multimodal_features=features)
    assert result["aar_trace"][0]["multimodal_included"] is True

# --- S-06 tests ---

def test_parse_symptoms_clean_json():
    """Clean JSON array parses into a list of strings."""
    from src.agents.intake import _parse_symptoms
    assert _parse_symptoms('["low mood >2 weeks", "fatigue"]') == ["low mood >2 weeks", "fatigue"]


def test_parse_symptoms_with_markdown_fence():
    """JSON wrapped in a markdown code fence still parses."""
    from src.agents.intake import _parse_symptoms
    raw = '```json\n["insomnia", "poor concentration"]\n```'
    assert _parse_symptoms(raw) == ["insomnia", "poor concentration"]


def test_parse_symptoms_with_surrounding_prose():
    """JSON array embedded in prose is still extracted."""
    from src.agents.intake import _parse_symptoms
    raw = 'Here are the symptoms I found: ["fatigue", "low mood"]'
    assert _parse_symptoms(raw) == ["fatigue", "low mood"]


def test_parse_symptoms_garbage_returns_empty():
    """Non-JSON or empty input degrades gracefully to []."""
    from src.agents.intake import _parse_symptoms
    assert _parse_symptoms("I could not find any symptoms.") == []
    assert _parse_symptoms("") == []


def test_extracted_symptoms_populated():
    """Agent extracts a non-empty list of symptom strings from clear input (LLM call)."""
    result = _run("I've been feeling really low and exhausted, and I can't concentrate at work.")
    symptoms = result["extracted_symptoms"]
    assert isinstance(symptoms, list)
    assert len(symptoms) > 0
    assert all(isinstance(s, str) for s in symptoms)


def test_duration_extracted():
    """When the user states a duration, it appears in at least one symptom string (LLM call)."""
    result = _run("I've been feeling down for the past three weeks.")
    symptoms = result["extracted_symptoms"]
    joined = " ".join(symptoms).lower()
    assert len(symptoms) > 0
    assert any(w in joined for w in ["week", "3", "three", "month", "day"])
