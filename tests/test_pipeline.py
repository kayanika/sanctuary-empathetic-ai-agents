"""
Tests the LangGraph pipeline runs end-to-end with stub agents.
Does not call the LLM — validates graph wiring and state flow only.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.graph import tao_graph


def test_normal_flow():
    initial_state = {
        "user_input": "I have been feeling sad and tired for the past few weeks.",
        "multimodal_features": {"speech_rate": 120, "pause_duration": 0.8, "valence": -0.4},
        "dialogue_history": [],
        "aar_trace": [],
    }
    result = tao_graph.invoke(initial_state)

    assert result["extracted_symptoms"], "Intake agent did not extract symptoms"
    assert result["diagnostic_hypothesis"], "Diagnostic agent did not produce a hypothesis"
    assert result["assessment_approved"] is True, "Oversight agent did not approve"
    assert result["recommended_intervention"], "Routing agent did not assign an intervention"
    assert len(result["aar_trace"]) == 4, f"Expected 4 AAR entries, got {len(result['aar_trace'])}"
    assert not result["crisis_escalation"], "False crisis escalation on normal input"
    print("Normal flow: PASSED")
    print(f"AAR trace entries: {[e['agent'] for e in result['aar_trace']]}")


def test_crisis_escalation():
    initial_state = {
        "user_input": "I have been having thoughts of killing myself.",
        "multimodal_features": {},
        "dialogue_history": [],
        "aar_trace": [],
    }
    result = tao_graph.invoke(initial_state)

    assert result["crisis_escalation"] is True, "Crisis not detected on high-risk input"
    assert "988" in result["recommended_intervention"] or "crisis" in result["recommended_intervention"].lower()
    print("Crisis escalation: PASSED")


if __name__ == "__main__":
    print("--- Testing normal flow ---")
    test_normal_flow()
    print("\n--- Testing crisis escalation ---")
    test_crisis_escalation()
    print("\nAll pipeline tests passed.")
