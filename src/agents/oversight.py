import os
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.pipeline.state import TAOState
from src.rag.retriever import retrieve

OVERSIGHT_SYSTEM_PROMPT = """You are an independent clinical oversight auditor. You review a diagnostic hypothesis produced by another agent and check it for specific error classes.

You are given: the extracted symptoms (the ONLY symptoms actually reported by the person), the diagnostic hypothesis under review, and retrieved DSM-5 criteria.

Check for these violations:
1. TEMPORAL (E1): the hypothesis asserts a diagnosis whose DSM-5 minimum duration is not met by the reported durations (major depressive episode >= 2 weeks, PTSD > 1 month, generalized anxiety >= 6 months).
2. FABRICATION (E2): the hypothesis cites or relies on a symptom that does NOT appear in the extracted symptoms list.
3. EXCLUSION (E4): the reported symptoms or context indicate a plausible alternative cause (e.g. substance use, another medical condition) that the hypothesis fails to rule out or acknowledge. If NO such alternative cause is indicated, this is NOT a violation.

Rules:
- The extracted symptoms list is authoritative. If the hypothesis mentions a symptom not in that list, that is fabrication.
- Only flag a violation you can justify from the provided material. Appropriate tentativeness is NOT a violation.
- Respond with ONLY a JSON object, no prose, no markdown:
  {"temporal_violation": <true|false>, "fabrication_violation": <true|false>, "exclusion_violation": <true|false>, "explanation": "<one short sentence describing any violations, or 'no violations found'>", "confidence": <0.0-1.0>}"""


def _parse_verdict(raw: str) -> dict:
    """Parse the auditor's JSON verdict. Falls back to a SAFE default (flag every
    class for review) if parsing fails — an unparseable verifier must never approve."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return {
            "temporal_violation": bool(data.get("temporal_violation", True)),
            "fabrication_violation": bool(data.get("fabrication_violation", True)),
            "exclusion_violation": bool(data.get("exclusion_violation", True)),
            "explanation": str(data.get("explanation", "")).strip(),
            "confidence": float(data.get("confidence", 0.0)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "temporal_violation": True,
            "fabrication_violation": True,
            "exclusion_violation": True,
            "explanation": "Verifier output could not be parsed; flagged for review.",
            "confidence": 0.0,
        }


VIOLATION_LABELS = {
    "temporal_violation": "Temporal threshold (E1)",
    "fabrication_violation": "Symptom fabrication (E2)",
    "exclusion_violation": "Exclusion omission (E4)",
}


def oversight_agent(state: TAOState) -> dict:
    """
    Agent 3 — Oversight & Auditing.
    Grounded LLM verifier checking the diagnostic hypothesis for temporal (E1),
    fabrication (E2), and exclusion (E4) violations. (S-10, S-11)
    """
    symptoms = state.get("extracted_symptoms", [])
    hypothesis = state.get("diagnostic_hypothesis", "")

    query = hypothesis if hypothesis else "; ".join(symptoms)
    criteria_passages = retrieve(query or "DSM-5 diagnostic criteria", k=4)

    symptoms_block = "\n".join(f"- {s}" for s in symptoms) if symptoms else "(none)"
    passages_block = "\n\n".join(criteria_passages)
    user_content = (
        f"Extracted symptoms (the ONLY symptoms reported):\n{symptoms_block}\n\n"
        f"Diagnostic hypothesis under review:\n{hypothesis}\n\n"
        f"Retrieved DSM-5 criteria:\n{passages_block}"
    )

    model = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:12b-mlx"),
        temperature=0.0,
    )
    response = model.invoke([
        SystemMessage(content=OVERSIGHT_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    verdict = _parse_verdict(response.content)

    corrections = [
        f"{label}: {verdict['explanation']}"
        for key, label in VIOLATION_LABELS.items()
        if verdict[key]
    ]
    approved = len(corrections) == 0

    trace_entry = {
        "agent": "oversight",
        "input_summary": f"Reviewing hypothesis (tier {state.get('phq_tier', 'unknown')})",
        "output_summary": "APPROVED" if approved else f"REJECTED — {verdict['explanation']}",
        "temporal_violation": verdict["temporal_violation"],
        "fabrication_violation": verdict["fabrication_violation"],
        "exclusion_violation": verdict["exclusion_violation"],
        "confidence": verdict["confidence"],
    }
    return {
        "audit_notes": verdict["explanation"],
        "assessment_approved": approved,
        "oversight_corrections": corrections,
        "aar_trace": [trace_entry],
        "oversight_confidence": verdict["confidence"],

    }
