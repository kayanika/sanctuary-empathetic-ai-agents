import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Single source of truth for the shared/default model. Agents 1, 2, 4, and
# Agent 3's same-base condition all resolve to this when they don't pass an
# explicit model — the same-base comparison depends on that being true.
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")


def get_chat_model(temperature: float = 0.0, *, model: str | None = None, provider: str | None = None):
    """Return a LangChain chat model. Agents normally only need to pass a model
    name (or nothing, to get the shared default); provider resolves globally.

    For now (Phase 1, synthetic-data-only) everything defaults to OpenRouter.
    Local Ollama is kept for Phase 3: once real DAIC-WOZ data is involved, the
    pipeline MUST run locally instead — its data use agreement precludes sending
    participant-derived content to a third-party API. Pass provider="ollama"
    explicitly (or set MODEL_PROVIDER=ollama) for any real-data run.
    """
    provider = (provider or os.getenv("MODEL_PROVIDER", "openrouter")).strip().lower()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set in .env to use OpenRouter")
        return ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            max_retries=5,
        )

    if provider == "ollama":
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model or os.getenv("OLLAMA_MODEL", "gemma4:12b-mlx"),
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider '{provider}' — expected 'openrouter' or 'ollama'")
