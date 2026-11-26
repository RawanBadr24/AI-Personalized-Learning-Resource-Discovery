"""LLM (Groq via OpenAILike) and embedding model factories."""
from __future__ import annotations

from functools import lru_cache

from learning_agent.config import config


@lru_cache(maxsize=1)
def get_llm():
    from llama_index.llms.openai_like import OpenAILike

    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OpenAILike(
        model=config.groq_model,
        api_base=config.groq_base_url,
        api_key=config.groq_api_key,
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=32768,
        temperature=config.temperature,
    )


@lru_cache(maxsize=1)
def get_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=config.embed_model)


def configure_llama_index() -> None:
    """Register the LLM + embeddings as LlamaIndex global defaults."""
    from llama_index.core import Settings

    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()
