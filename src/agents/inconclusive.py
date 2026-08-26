from src.pipeline.state import TAOState

INCONCLUSIVE_MESSAGE = (
    "This screening could not be completed with confidence. The automated review "
    "flagged concerns with the preliminary assessment, so no recommendation is being "
    "offered. Please consider speaking with a qualified mental health professional."
)


def inconclusive_agent(state: TAOState) -> dict:
    """Terminal node for rejected assessments. Ensures a rejection still produces a
    safe, honest output rather than silently ending the pipeline."""
    corrections = state.get("oversight_corrections", [])
    trace_entry = {
        "agent": "inconclusive",
        "input_summary": f"Oversight rejected with {len(corrections)} correction(s)",
        "output_summary": "Safe inconclusive output; no intervention recommended.",
    }
    return {
        "recommended_intervention": INCONCLUSIVE_MESSAGE,
        "aar_trace": [trace_entry],
    }
