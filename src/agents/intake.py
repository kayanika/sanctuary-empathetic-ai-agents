from src.pipeline.state import TAOState


def intake_agent(state: TAOState) -> dict:
    """
    Agent 1 — Multimodal Intake
    Engages the user, processes dialogue alongside multimodal features,
    and extracts reported symptoms for the Diagnostic Agent.
    (Stub: real LLM logic added in Phase 4)
    """
    trace_entry = {
        "agent": "intake",
        "input_summary": state.get("user_input", ""),
        "output_summary": "Stub: symptoms extracted",
    }
    return {
        "dialogue_history": [{"role": "user", "content": state.get("user_input", "")}],
        "extracted_symptoms": ["[stub symptom 1]", "[stub symptom 2]"],
        "aar_trace": [trace_entry],
    }
