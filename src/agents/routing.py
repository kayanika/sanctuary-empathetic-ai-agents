from src.pipeline.state import TAOState
from src.safety.crisis import check_crisis


def routing_agent(state: TAOState) -> dict:
    """
    Agent 4 — JITAI Routing
    Maps the verified assessment to a personalized micro-intervention.
    Hard-coded crisis escalation triggers if acute risk keywords are detected —
    this check is architectural and cannot be overridden by the LLM.
    (Stub: real JITAI mapping logic added in Phase 7)
    """
    user_input = state.get("user_input", "").lower()
    crisis = check_crisis(state.get("user_input", ""))


    trace_entry = {
        "agent": "routing",
        "input_summary": f"PHQ tier: {state.get('phq_tier', 'unknown')}",
        "output_summary": "CRISIS ESCALATION" if crisis else "Stub: intervention assigned",
    }
    return {
        "crisis_escalation": crisis,
        "recommended_intervention": (
            "CRISIS: Immediate referral to human crisis line (e.g. 988 Lifeline)."
            if crisis
            else "[Stub] Low-arousal auditory grounding exercise recommended."
        ),
        "aar_trace": [trace_entry],
    }
