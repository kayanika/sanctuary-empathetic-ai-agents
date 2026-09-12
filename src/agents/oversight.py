import os
import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.pipeline.state import TAOState
from src.pipeline.llm_factory import get_chat_model, DEFAULT_MODEL
from src.rag.retriever import retrieve

# Agent 2 generates with DEFAULT_MODEL, so a verifier using it is "same-base" —
# the self-preference condition H1 tests.
GENERATOR_MODEL = DEFAULT_MODEL
ALT_VERIFIER_MODEL = os.getenv("ALT_VERIFIER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")


@dataclass(frozen=True)
class VerifierConfig:
    """One cell of the S-25 configuration grid."""
    model: str = GENERATOR_MODEL
    grounded: bool = True
    temperature: float = 0.0
    persona: str = "neutral"          # "neutral" | "skeptic"

    @property
    def base_pairing(self) -> str:
        return "same-base" if self.model == GENERATOR_MODEL else "mixed-base"

    @property
    def label(self) -> str:
        grounding = "grounded" if self.grounded else "ungrounded"
        return f"{self.base_pairing}|{grounding}|t{self.temperature}|{self.persona}"


BASE_PROMPT = """You are an independent clinical oversight auditor. You review a diagnostic hypothesis produced by another agent and check it for specific error classes.

You are given: the extracted symptoms (the ONLY symptoms actually reported by the person), the diagnostic hypothesis under review, and DSM-5 criteria.

Check for these violations:
1. TEMPORAL (E1): the hypothesis asserts a diagnosis whose DSM-5 minimum duration is not met by the reported durations (major depressive episode >= 2 weeks, PTSD > 1 month, generalized anxiety >= 6 months).
2. FABRICATION (E2): the hypothesis cites or relies on a symptom that does NOT appear in the extracted symptoms list.
3. EXCLUSION (E4): the reported symptoms or context indicate a plausible alternative cause (e.g. substance use, another medical condition) that the hypothesis fails to rule out or acknowledge. If NO such alternative cause is indicated, this is NOT a violation.

The extracted symptoms list is authoritative. If the hypothesis mentions a symptom not in that list, that is fabrication."""

PERSONA_SUFFIX = {
    "neutral": (
        "\n\nReview impartially. Flag only violations you can justify from the material "
        "provided; appropriate tentativeness is NOT a violation."
    ),
    "skeptic": (
        "\n\nAdopt an adversarial, skeptical stance. Assume an error may well be present "
        "and actively hunt for it. Do not give the hypothesis the benefit of the doubt."
    ),
}

OUTPUT_CONTRACT = """

Respond with ONLY a JSON object, no prose, no markdown:
  {"temporal_violation": <true|false>, "fabrication_violation": <true|false>, "exclusion_violation": <true|false>, "explanation": "<one short sentence describing any violations, or 'no violations found'>", "confidence": <0.0-1.0>}"""


def build_prompt(config: VerifierConfig) -> str:
    return BASE_PROMPT + PERSONA_SUFFIX[config.persona] + OUTPUT_CONTRACT


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


def verify(state, config: VerifierConfig | None = None) -> dict:
    """Configurable verifier core. `oversight_agent` wraps this with the default
    config; the S-25 grid runner varies `config` across all cells."""
    config = config or VerifierConfig()
    symptoms = state.get("extracted_symptoms", [])
    hypothesis = state.get("diagnostic_hypothesis", "")

    if config.grounded:
        query = hypothesis or "; ".join(symptoms) or "DSM-5 diagnostic criteria"
        criteria_block = "\n\n".join(retrieve(query, k=4))
    else:
        criteria_block = "(no criteria retrieved — rely on your own knowledge)"

    symptoms_block = "\n".join(f"- {s}" for s in symptoms) if symptoms else "(none)"
    user_content = (
        f"Extracted symptoms (the ONLY symptoms reported):\n{symptoms_block}\n\n"
        f"Diagnostic hypothesis under review:\n{hypothesis}\n\n"
        f"DSM-5 criteria:\n{criteria_block}"
    )

    model = get_chat_model(temperature=config.temperature, model=config.model)
    response = model.invoke([
        SystemMessage(content=build_prompt(config)),
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
        "verifier_config": config.label,
    }
    return {
        "audit_notes": verdict["explanation"],
        "assessment_approved": approved,
        "oversight_corrections": corrections,
        "oversight_confidence": verdict["confidence"],
        "aar_trace": [trace_entry],
    }


def oversight_agent(state: TAOState) -> dict:
    """Agent 3 — LangGraph node. Uses the default verifier configuration
    (same-base, grounded, temp 0, neutral persona)."""
    return verify(state, VerifierConfig())
