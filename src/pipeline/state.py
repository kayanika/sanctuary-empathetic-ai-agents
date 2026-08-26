from typing import TypedDict, Annotated
import operator


class TAOState(TypedDict):
    # --- Input ---
    user_input: str
    multimodal_features: dict  # pre-extracted audio/AU features as JSON payload

    # --- Agent 1: Intake ---
    dialogue_history: Annotated[list[dict], operator.add]  # accumulates across turns
    extracted_symptoms: list[str]

    # --- Agent 2: Diagnostic ---
    diagnostic_hypothesis: str
    retrieved_passages: list[str]   # DSM-5 passages retrieved via RAG
    phq_tier: str  # "minimal" | "mild" | "moderate" | "severe" | "undetermined"


    # --- Agent 3: Oversight ---
    audit_notes: str
    assessment_approved: bool
    oversight_corrections: list[str]
    oversight_confidence: float     # verifier's self-reported confidence (for S-26 calibration)


    # --- Agent 4: Routing ---
    recommended_intervention: str
    crisis_escalation: bool

    # --- AAR Trace (used in XAI / Prolific study) ---
    # Every inter-agent handoff is logged here for the After-Action Review
    aar_trace: Annotated[list[dict], operator.add]
