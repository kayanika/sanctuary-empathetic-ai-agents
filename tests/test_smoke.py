"""
Smoke test: verifies Ollama + Gemma 4 and local embeddings are working
before any agent code is built.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb


def test_llm():
    llm = ChatOllama(model="gemma4:12b-mlx", temperature=0)
    response = llm.invoke([HumanMessage(content="Reply with one word: ready")])
    assert response.content.strip(), "LLM returned empty response"
    print(f"LLM response: {response.content.strip()}")


def test_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector = embeddings.embed_query("test sentence")
    assert len(vector) == 384, f"Unexpected embedding dimension: {len(vector)}"
    print(f"Embedding dimension: {len(vector)} — OK")


def test_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("smoke_test")
    collection.add(documents=["hello world"], ids=["1"])
    result = collection.query(query_texts=["hello"], n_results=1)
    assert result["documents"][0][0] == "hello world"
    print("ChromaDB read/write — OK")
    client.delete_collection("smoke_test")


if __name__ == "__main__":
    print("--- Testing LLM (this may take ~30s on first call) ---")
    test_llm()
    print("\n--- Testing embeddings (downloads ~90MB on first run) ---")
    test_embeddings()
    print("\n--- Testing ChromaDB ---")
    test_chromadb()
    print("\nAll checks passed. Ready to build.")
