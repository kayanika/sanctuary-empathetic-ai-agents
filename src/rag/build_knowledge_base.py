"""
    
    S-07 - DSM-5 Knowledge Base builder.
    
    Builds a BhromaDB collection of diagnostic criteria for Agent 2's RAG retrieval.
    
    Copyright-safe design (Mock Fallback pattern):
    - Looks first for a gitignored real life : data/dsm5_real.txt
    - If absent, warns about the copyright restriction and falls back to the committed synthetic file: data/dsm5_mpck.txt
The synthetic file is paraphrase and structurally faithful, NOT verbatim DSM-5
"""

import os
from pathlib import Path
from langchain_core.documents import Document

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()

DATA_DIR = Path("data")
REAL_FILE = DATA_DIR/ "dsm5_real.txt"
MOCK_FILE = DATA_DIR / "dsm5_mock.txt"

CHROMA_PATH = os.getenv("CHROMA_DB_PATH","./data/chroma_db")
COLLECTION_NAME = "dsm5_criteria"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _select_source() -> Path:
    if REAL_FILE.exists():
        print(f"[knowledge base] Using licensed real criteria: {REAL_FILE}")
        return REAL_FILE
    print(
        f"[knowledge base] WARNING: {REAL_FILE} not found.\n"
        "  Verbatim DSM-5 is copyrighted (APA) and is NOT committed to this repo.\n"
        "  Place a licensed copy at data/dsm5_real.txt (gitignored) to use real criteria.\n"
        f"  Falling back to synthetic, paraphrased criteria: {MOCK_FILE}"
    )
    if not MOCK_FILE.exists():
        raise FileNotFoundError(
            f"Neither {REAL_FILE} nor {MOCK_FILE} exists; cannot build knowledge base."
        )
    return MOCK_FILE

def build_knowledge_base() -> Chroma:
    source = _select_source()
    text = source.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"source": source.name, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    print(f"[knowledge base] Split into {len(documents)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # Reset the collection so re-runs don't duplicate chunks (idempotent).
    Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    print(f"[knowledge base] Stored in {CHROMA_PATH} (collection '{COLLECTION_NAME}').")
    return vectorstore
    
if __name__ == "__main__":
    vs = build_knowledge_base()
    results = vs.similarity_search(
        "How long must low mood last to meet major depression?", k=2
    )
    print("\n[knowledge base] Sample retrieval for a duration query:")
    for r in results:
        print(f"  - {r.page_content[:120]}...")