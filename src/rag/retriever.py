"""
Retriever for Agent 2 — loads the persisted DSM-5 ChromaDB collection built by
build_knowledge_base.py and returns the top-k most relevant criteria passages.

This does NOT rebuild the index; it reads the already-persisted store on disk.
"""
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.build_knowledge_base import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL

load_dotenv()

_vectorstore = None  # module-level cache so we load the model/store only once


def _get_vectorstore() -> Chroma:
    """Lazily load the persisted collection once and reuse it on later calls."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
    return _vectorstore


def retrieve(query: str, k: int = 4) -> list[str]:
    """Return the text of the top-k DSM-5 passages most relevant to the query."""
    vs = _get_vectorstore()
    results = vs.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
