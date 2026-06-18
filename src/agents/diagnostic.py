from src.pipeline.state import TAOState


def diagnostic_agent(state: TAOState) -> dict:
    """
    Agent 2 — Diagnostic Retrieval
    Uses RAG to retrieve relevant DSM-5 passages and cross-references them
    against the extracted symptoms to form a diagnostic hypothesis.
    (Stub: real RAG + LLM logic added in Phase 5)
    """
    trace_entry = {
        "agent": "diagnostic",
        "input_summary": f"Symptoms received: {state.get('extracted_symptoms', [])}",
        "output_summary": "Stub: diagnostic hypothesis formed",
    }
    return {
        "diagnostic_hypothesis": "[Stub] Possible mild depressive episode based on reported symptoms.",
        "retrieved_passages": ["[Stub DSM-5 passage 1]", "[Stub DSM-5 passage 2]"],
        "phq_tier": "mild",
        "aar_trace": [trace_entry],
    }
