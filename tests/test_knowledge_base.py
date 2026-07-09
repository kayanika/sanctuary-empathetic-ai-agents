"""S-07 acceptance test — DSM-5 knowledge base retrieval.

No LLM calls (embeddings only), so this is fast and low-heat. The embedding model
downloads once on first run, then is cached.
"""
from src.rag.build_knowledge_base import build_knowledge_base


def test_depression_query_returns_depression_content():
    vs = build_knowledge_base()
    results = vs.similarity_search("persistent low mood and loss of interest", k=3)
    assert len(results) > 0
    assert any("depress" in d.page_content.lower() for d in results)


def test_ptsd_query_returns_trauma_content():
    vs = build_knowledge_base()
    results = vs.similarity_search("flashbacks and nightmares after a traumatic event", k=3)
    assert len(results) > 0
    joined = " ".join(d.page_content.lower() for d in results)
    assert "trauma" in joined or "ptsd" in joined or "stress" in joined


def test_mdd_duration_threshold_is_retrievable():
    """The 2-week MDD threshold (which Agent 3 checks) must be retrievable."""
    vs = build_knowledge_base()
    results = vs.similarity_search("minimum duration for a major depressive episode", k=3)
    joined = " ".join(d.page_content.lower() for d in results)
    assert "two week" in joined or "2 week" in joined or "14 day" in joined
