"""S-23 tests — error injection harness produces exactly one known error per variant."""
import pytest

from src.evaluation.base_cases import load_base_cases
from src.evaluation.injection import (
    inject, build_variants, make_clean_control, applicable_errors, ERROR_TYPES,
)


def _case(case_id):
    return next(c for c in load_base_cases() if c["case_id"] == case_id)


def test_originals_are_never_mutated():
    """Ground truth must survive injection untouched."""
    case = _case("MDD-002")
    before = str(case)
    for error_type in applicable_errors(case):
        inject(case, error_type)
    assert str(case) == before


def test_injection_is_deterministic():
    case = _case("MDD-002")
    assert inject(case, "E2") == inject(case, "E2")


def test_e1_shortens_durations_but_keeps_hypothesis():
    case = _case("MDD-002")
    v = inject(case, "E1")
    assert "4 days" in " ".join(v["extracted_symptoms"])
    assert v["diagnostic_hypothesis"] == case["diagnostic_hypothesis"]


def test_e1_not_applicable_to_subthreshold():
    case = _case("MDD-004")
    assert "E1" not in applicable_errors(case)
    with pytest.raises(ValueError, match="not applicable"):
        inject(case, "E1")


def test_e2_cites_unreported_symptoms():
    case = _case("MDD-002")
    v = inject(case, "E2")
    assert len(v["diagnostic_hypothesis"]) > len(case["diagnostic_hypothesis"])
    assert v["extracted_symptoms"] == case["extracted_symptoms"]


def test_e3_shifts_tier_materially():
    case = _case("MDD-002")   # moderate
    v = inject(case, "E3")
    assert v["phq_tier"] == "minimal"
    assert v["extracted_symptoms"] == case["extracted_symptoms"]


def test_e4_adds_alternative_cause():
    case = _case("MDD-002")
    v = inject(case, "E4")
    assert "alcohol" in " ".join(v["extracted_symptoms"]).lower()
    assert v["diagnostic_hypothesis"] == case["diagnostic_hypothesis"]


def test_e5_adds_ideation_and_flags_crisis():
    case = _case("MDD-002")
    v = inject(case, "E5")
    assert v["crisis_expected"] is True
    assert "ending my life" in v["user_input"].lower()
    assert v["diagnostic_hypothesis"] == case["diagnostic_hypothesis"]


def test_clean_control_expects_no_violation():
    control = make_clean_control(_case("MDD-002"))
    assert control["expected_violation"] is False
    assert control["error_type"] is None


def test_build_variants_shape():
    variants = build_variants()
    assert len(variants) == 45          # 8 clean + 5 E1 + 8 each for E2-E5
    trace_ids = [v["trace_id"] for v in variants]
    assert len(trace_ids) == len(set(trace_ids))
    clean = [v for v in variants if v["error_type"] is None]
    assert len(clean) == 8


def test_every_error_type_represented():
    seen = {v["error_type"] for v in build_variants()}
    assert set(ERROR_TYPES).issubset(seen)
