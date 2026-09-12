"""S-27 tests — configurable verifier + heterogeneous model support.

Both same-base and mixed-base conditions now call OpenRouter, so live tests need
OPENROUTER_API_KEY and are skipped without it — this avoids burning free-tier
quota on every routine test run.
"""
import os
import pytest
from dotenv import load_dotenv
load_dotenv()

from src.agents.oversight import VerifierConfig, build_prompt, verify, GENERATOR_MODEL, ALT_VERIFIER_MODEL


# --- config: no network ---

def test_default_config_is_same_base():
    assert VerifierConfig().base_pairing == "same-base"
    assert VerifierConfig().model == GENERATOR_MODEL


def test_alt_model_is_mixed_base():
    assert VerifierConfig(model=ALT_VERIFIER_MODEL).base_pairing == "mixed-base"


def test_label_encodes_all_four_dimensions():
    label = VerifierConfig(model=ALT_VERIFIER_MODEL, grounded=False,
                           temperature=0.7, persona="skeptic").label
    assert label == "mixed-base|ungrounded|t0.7|skeptic"


def test_persona_changes_prompt():
    neutral = build_prompt(VerifierConfig(persona="neutral"))
    skeptic = build_prompt(VerifierConfig(persona="skeptic"))
    assert neutral != skeptic
    assert "adversarial" in skeptic.lower()


# --- live calls: both conditions go through OpenRouter now ---

_SAMPLE_STATE = {
    "extracted_symptoms": ["low mood >3 weeks", "fatigue >3 weeks"],
    "diagnostic_hypothesis": "The patient reports insomnia and weight loss, meeting full criteria for severe major depressive disorder.",
    "phq_tier": "severe",
}


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_same_base_produces_verdict_shape():
    result = verify(_SAMPLE_STATE, VerifierConfig())
    assert isinstance(result["assessment_approved"], bool)
    assert result["aar_trace"][0]["verifier_config"].startswith("same-base")


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_mixed_base_produces_verdict_shape():
    result = verify(_SAMPLE_STATE, VerifierConfig(model=ALT_VERIFIER_MODEL))
    assert isinstance(result["assessment_approved"], bool)
    assert result["aar_trace"][0]["verifier_config"].startswith("mixed-base")
