"""S-08 tests — RAG-augmented diagnostic reasoning.

Assumes the DSM-5 knowledge base is already built:
    python -m src.rag.build_knowledge_base
The retrieve test uses embeddings only (fast, low heat); the agent tests each
make one Ollama call.
"""
from dotenv import load_dotenv
load_dotenv()

from src.agents.diagnostic import diagnostic_agent
from src.rag.retriever import retrieve


def _run(symptoms, user_input="") -> dict:
    state = {
        "user_input": user_input,
        "multimodal_features": {},
        "dialogue_history": [],
        "extracted_symptoms": symptoms,
        "diagnostic_hypothesis": "",
        "retrieved_passages": [],
        "phq_tier": "",
        "audit_notes": "",
        "assessment_approved": False,
        "oversight_corrections": [],
        "recommended_intervention": "",
        "crisis_escalation": False,
        "aar_trace": [],
    }
    return diagnostic_agent(state)


# --- retriever: embeddings only, no LLM ---

def test_retrieve_returns_relevant_passages():
    passages = retrieve("persistent low mood and loss of interest for two weeks", k=4)
    assert isinstance(passages, list)
    assert len(passages) > 0
    assert all(isinstance(p, str) for p in passages)
    assert any("depress" in p.lower() for p in passages)


# --- diagnostic agent: one LLM call each ---

def test_diagnostic_populates_state():
    result = _run(["low mood >2 weeks", "fatigue", "poor concentration"])
    assert len(result["retrieved_passages"]) > 0
    assert isinstance(result["diagnostic_hypothesis"], str)
    assert len(result["diagnostic_hypothesis"].strip()) > 20


def test_diagnostic_trace_records_retrieval():
    result = _run(["low mood >2 weeks", "anhedonia"])
    assert result["aar_trace"][0]["agent"] == "diagnostic"
    assert result["aar_trace"][0]["num_passages_retrieved"] > 0


def test_diagnostic_handles_no_symptoms():
    """Empty symptom list must not crash; it falls back to the user_input query."""
    result = _run([], user_input="I have been feeling really down and hopeless lately")
    assert isinstance(result["diagnostic_hypothesis"], str)
    assert isinstance(result["retrieved_passages"], list)

# --- S-09: PHQ tier classification ---

def test_parse_tier_valid_value():
    from src.agents.diagnostic import _parse_tier
    assert _parse_tier("moderate") == "moderate"
    assert _parse_tier("SEVERE") == "severe"


def test_parse_tier_with_surrounding_text():
    from src.agents.diagnostic import _parse_tier
    assert _parse_tier("The severity tier is mild.") == "mild"


def test_parse_tier_ambiguous_prefers_more_severe():
    from src.agents.diagnostic import _parse_tier
    assert _parse_tier("somewhere between moderate and severe") == "severe"


def test_parse_tier_garbage_returns_undetermined():
    from src.agents.diagnostic import _parse_tier
    assert _parse_tier("I cannot tell") == "undetermined"
    assert _parse_tier("") == "undetermined"


def test_tier_is_always_a_valid_value():
    """The agent's phq_tier must always be one of the allowed values (LLM call)."""
    result = _run(["low mood >2 weeks", "fatigue", "worthlessness", "insomnia", "poor concentration"])
    assert result["phq_tier"] in {"minimal", "mild", "moderate", "severe", "undetermined"}
