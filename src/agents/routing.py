from src.pipeline.state import TAOState

CRISIS_KEYWORDS = [
    "suicidal", "suicide",
    "self-harm", "self harm", "selfharm",
    "kill myself", "killing myself", "killed myself",
    "end my life", "ending my life",
    "hurt myself", "hurting myself",
    "want to die", "wanting to die",
    "don't want to live", "dont want to live",
    "no reason to live",
]


def routing_agent(state: TAOState) -> dict:
    """
    Agent 4 — JITAI Routing
    Maps the verified assessment to a personalized micro-intervention.
    Hard-coded crisis escalation triggers if acute risk keywords are detected —
    this check is architectural and cannot be overridden by the LLM.
    (Stub: real JITAI mapping logic added in Phase 7)
    """
    user_input = state.get("user_input", "").lower()
    crisis = any(kw in user_input for kw in CRISIS_KEYWORDS)

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
