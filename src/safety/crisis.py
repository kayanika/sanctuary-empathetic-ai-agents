"""
Deterministic crisis detection — the architectural safety gate.

This is code-level, not LLM-based: it runs on raw user input before any model is
invoked and cannot be overridden by model behaviour. It is deliberately
high-recall — false positives are acceptable here, false negatives are not.
"""

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

CRISIS_MESSAGE = (
    "It sounds like you may be going through something extremely difficult, and you "
    "deserve support from a person right now. Please contact a crisis line: in the US, "
    "call or text 988 (Suicide & Crisis Lifeline). If you are in immediate danger, "
    "call your local emergency number."
)


def check_crisis(text: str) -> bool:
    """Return True if the raw text contains any acute-risk indicator."""
    lowered = (text or "").lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def crisis_gate(state) -> dict:
    """First node in the graph. Runs on raw user input BEFORE any LLM call, so the
    escalation path cannot be bypassed by downstream routing decisions."""
    crisis = check_crisis(state.get("user_input", ""))
    trace_entry = {
        "agent": "crisis_gate",
        "input_summary": state.get("user_input", "")[:300],
        "output_summary": "CRISIS ESCALATION" if crisis else "No acute risk detected",
    }
    result = {"crisis_escalation": crisis, "aar_trace": [trace_entry]}
    if crisis:
        result["recommended_intervention"] = CRISIS_MESSAGE
    return result
