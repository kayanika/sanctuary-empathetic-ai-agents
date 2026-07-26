import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.pipeline.state import TAOState
from src.rag.retriever import retrieve

VALID_TIERS = {"minimal", "mild", "moderate", "severe"}

DIAGNOSTIC_SYSTEM_PROMPT = """You are a clinical reasoning assistant performing preliminary, non-final analysis. You are given (1) a list of symptoms extracted from a screening conversation and (2) retrieved diagnostic criteria passages.

Rules:
- Reason ONLY from the provided symptoms and the retrieved criteria. Do NOT invent symptoms that were not reported.
- Explicitly cite which retrieved criteria the symptoms match, including any duration thresholds (e.g. the 2-week minimum for a major depressive episode).
- State clearly which criteria are NOT met or where evidence is insufficient, rather than overstating.
- Output 3-5 sentences: the most plausible screening hypothesis, the specific criteria it rests on, and any criteria not met.
- This is a screening hypothesis, not a diagnosis."""

PHQ_TIER_SYSTEM_PROMPT = """You are a screening severity classifier. Given a list of reported symptoms and a diagnostic hypothesis, assign a single PHQ-8 severity tier.

Tiers (aligned to PHQ-8 total-score bands):
- minimal  (PHQ-8 roughly 0-4): few or no depressive symptoms
- mild     (PHQ-8 roughly 5-9)
- moderate (PHQ-8 roughly 10-14)
- severe   (PHQ-8 roughly 15-24)

Rules:
- Base the tier on the number and severity of the reported symptoms.
- Output EXACTLY ONE word: minimal, mild, moderate, or severe. No punctuation, no explanation.
- If there is genuinely not enough information to judge, output: undetermined"""


def _parse_tier(raw: str) -> str:
    """Return one of the four valid tiers, or 'undetermined' as a safe default.
    Never silently downgrades an uncertain case to 'minimal'; ambiguous cases
    (e.g. 'moderate to severe') resolve to the MORE severe tier."""
    text = raw.strip().lower()
    if text in VALID_TIERS:
        return text
    for tier in ("severe", "moderate", "mild", "minimal"):
        if tier in text:
            return tier
    return "undetermined"


def diagnostic_agent(state: TAOState) -> dict:
    """
    Agent 2 — Diagnostic Retrieval (RAG).
    Retrieves relevant DSM-5 criteria for the extracted symptoms, reasons over them
    to form a grounded diagnostic hypothesis (S-08), and assigns a PHQ severity
    tier via a separate deterministic call (S-09).
    """
    symptoms = state.get("extracted_symptoms", [])
    query = "; ".join(symptoms) if symptoms else state.get("user_input", "")

    retrieved_passages = retrieve(query, k=4)

    symptoms_block = (
        "\n".join(f"- {s}" for s in symptoms)
        if symptoms else "(no structured symptoms were extracted)"
    )
    passages_block = "\n\n".join(
        f"[Criterion {i + 1}]\n{p}" for i, p in enumerate(retrieved_passages)
    )
    user_content = (
        f"Extracted symptoms:\n{symptoms_block}\n\n"
        f"Retrieved diagnostic criteria:\n{passages_block}"
    )

    # --- Diagnostic hypothesis: grounded reasoning (temp 0.3) — S-08 ---
    model = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:12b-mlx"),
        temperature=0.3,
    )
    response = model.invoke([
        SystemMessage(content=DIAGNOSTIC_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    hypothesis = response.content

    # --- PHQ severity tier: deterministic classification (temp 0.0) — S-09 ---
    tier_model = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:12b-mlx"),
        temperature=0.0,
    )
    tier_response = tier_model.invoke([
        SystemMessage(content=PHQ_TIER_SYSTEM_PROMPT),
        HumanMessage(content=f"Symptoms:\n{symptoms_block}\n\nHypothesis:\n{hypothesis}"),
    ])
    phq_tier = _parse_tier(tier_response.content)

    trace_entry = {
        "agent": "diagnostic",
        "input_summary": f"Symptoms: {symptoms}",
        "output_summary": hypothesis[:300],
        "num_passages_retrieved": len(retrieved_passages),
        "phq_tier": phq_tier,
    }
    return {
        "diagnostic_hypothesis": hypothesis,
        "retrieved_passages": retrieved_passages,
        "phq_tier": phq_tier,
        "aar_trace": [trace_entry],
    }
