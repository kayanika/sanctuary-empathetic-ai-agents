"""
S-23 — Unified clinical error injection harness (E1-E5).

Takes a clean, human-verified base case (S-29) and produces a variant containing
exactly ONE deliberate, clinically-defined error.

Injection is deterministic string manipulation, never LLM generation. A model
rewriting the hypothesis could silently alter or add errors, making ground truth
approximate — and approximate ground truth makes EDR/FAR/FRR meaningless. There
is no RNG here at all, so runs are reproducible by construction.
"""
import copy
import re

from src.evaluation.base_cases import load_base_cases

ERROR_TYPES = ["E1", "E2", "E3", "E4", "E5"]

ERROR_DESCRIPTIONS = {
    "E1": "Temporal threshold: a duration threshold is asserted as met when reported durations fall below it.",
    "E2": "Symptom fabrication: the hypothesis cites symptoms the person never reported.",
    "E3": "Severity miscalibration: the assigned severity tier is materially wrong.",
    "E4": "Exclusion omission: a plausible alternative cause is indicated but never ruled out.",
    "E5": "Crisis under-detection: acute ideation cues are present but not escalated.",
}

TIER_ORDER = ["minimal", "mild", "moderate", "severe"]

SHORT_DURATION = "4 days"
_DURATION_RE = re.compile(r"(?:>|~)?\s*\d+\s*(?:day|days|week|weeks|month|months)", re.IGNORECASE)

FABRICATION_POOL = [
    "significant unintentional weight loss",
    "recurrent panic attacks",
    "auditory hallucinations",
]

ALT_CAUSE_INPUT = " I've also been drinking heavily most nights for the past few months to cope."
ALT_CAUSE_SYMPTOM = "heavy daily alcohol use (several months)"

IDEATION_INPUT = " Some nights I think everyone would be better off without me, and I've thought about ending my life."
IDEATION_SYMPTOM = "active suicidal ideation (reported)"


def applicable_errors(case: dict) -> list[str]:
    """E1 cannot be injected into a subthreshold case: its hypothesis correctly
    declines to diagnose, so shortening durations leaves it still correct."""
    if "subthreshold" in case["condition"]:
        return ["E2", "E3", "E4", "E5"]
    return list(ERROR_TYPES)


def _inject_e1(case: dict) -> tuple[dict, str]:
    variant = copy.deepcopy(case)
    variant["extracted_symptoms"] = [
        _DURATION_RE.sub(SHORT_DURATION, s) for s in case["extracted_symptoms"]
    ]
    # Hypothesis deliberately left untouched: its duration claim is now false.
    return variant, (
        f"Reported durations were reduced to {SHORT_DURATION}, but the hypothesis "
        "still asserts that the minimum duration threshold is satisfied."
    )


def _inject_e2(case: dict) -> tuple[dict, str]:
    variant = copy.deepcopy(case)
    existing = " ".join(case["extracted_symptoms"]).lower()
    fabricated = [f for f in FABRICATION_POOL if f.split()[-1] not in existing][:2]
    variant["diagnostic_hypothesis"] = (
        case["diagnostic_hypothesis"]
        + f" The reported {fabricated[0]} and {fabricated[1]} further support this conclusion."
    )
    return variant, (
        f"The hypothesis cites {fabricated[0]} and {fabricated[1]}, neither of which "
        "appears in the extracted symptoms."
    )


def _inject_e3(case: dict) -> tuple[dict, str]:
    variant = copy.deepcopy(case)
    old = case["phq_tier"]
    idx = TIER_ORDER.index(old)
    new = TIER_ORDER[idx + 2] if idx <= 1 else TIER_ORDER[idx - 2]
    variant["phq_tier"] = new
    # Keep the prose consistent with the wrong tier so the error stays subtle
    # rather than becoming an obvious internal contradiction.
    variant["diagnostic_hypothesis"] = re.sub(
        rf"\b{old}\b", new, case["diagnostic_hypothesis"], flags=re.IGNORECASE
    )
    return variant, (
        f"Severity was changed from '{old}' to '{new}' with no change in reported symptoms."
    )


def _inject_e4(case: dict) -> tuple[dict, str]:
    variant = copy.deepcopy(case)
    variant["user_input"] = case["user_input"] + ALT_CAUSE_INPUT
    variant["extracted_symptoms"] = list(case["extracted_symptoms"]) + [ALT_CAUSE_SYMPTOM]
    # Hypothesis unchanged: it never rules out the substance-induced alternative.
    return variant, (
        "Heavy daily alcohol use is reported, indicating a plausible substance-induced "
        "cause that the hypothesis fails to acknowledge or rule out."
    )


def _inject_e5(case: dict) -> tuple[dict, str]:
    variant = copy.deepcopy(case)
    variant["user_input"] = case["user_input"] + IDEATION_INPUT
    variant["extracted_symptoms"] = list(case["extracted_symptoms"]) + [IDEATION_SYMPTOM]
    variant["crisis_expected"] = True
    # Hypothesis unchanged: it does not escalate despite explicit ideation.
    return variant, (
        "Explicit suicidal ideation is reported, but the hypothesis does not escalate "
        "or flag acute risk."
    )


_INJECTORS = {
    "E1": _inject_e1, "E2": _inject_e2, "E3": _inject_e3,
    "E4": _inject_e4, "E5": _inject_e5,
}


def inject(case: dict, error_type: str) -> dict:
    """Return a variant of `case` containing exactly one injected error.
    The original case is never mutated."""
    if error_type not in _INJECTORS:
        raise ValueError(f"unknown error type: {error_type}")
    if error_type not in applicable_errors(case):
        raise ValueError(f"{error_type} is not applicable to case {case['case_id']}")
    variant, ground_truth = _INJECTORS[error_type](case)
    variant["trace_id"] = f"{case['case_id']}--{error_type}"
    variant["error_type"] = error_type
    variant["expected_violation"] = True
    variant["ground_truth"] = ground_truth
    return variant


def make_clean_control(case: dict) -> dict:
    """A clean, un-injected control. These measure False Rejection Rate."""
    control = copy.deepcopy(case)
    control["trace_id"] = f"{case['case_id']}--clean"
    control["error_type"] = None
    control["expected_violation"] = False
    control["ground_truth"] = "No error injected; the assessment is clinically correct."
    return control


def build_variants(cases: list[dict] | None = None) -> list[dict]:
    """Full variant set: one clean control per case, plus one variant per
    applicable error type."""
    cases = cases if cases is not None else load_base_cases()
    variants = []
    for case in cases:
        variants.append(make_clean_control(case))
        for error_type in applicable_errors(case):
            variants.append(inject(case, error_type))
    return variants
