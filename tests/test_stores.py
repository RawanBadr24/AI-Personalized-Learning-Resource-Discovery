"""Needs chromadb (and llama-index for ResourceStore). Skipped automatically if not installed."""
import hashlib

import pytest

pytest.importorskip("chromadb")


def fake_embed(text: str):
    """Deterministic 8-dim vector, so no model download is needed."""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in h[:8]]


def test_profile_store_roundtrip(tmp_path):
    from learning_agent.rag.store import ProfileStore

    store = ProfileStore(tmp_path / "db", embed_fn=fake_embed)
    assert store.get("nobody") == {}
    saved = store.save({"user_id": "u1", "subject": "RAG", "topics": ["a"]})
    assert "saved_at" in saved
    assert store.get("u1")["subject"] == "RAG"
    store.save({"user_id": "u1", "subject": "Updated"})  # upsert
    assert store.get("u1")["subject"] == "Updated"
    assert store.similar("RAG", k=3)[0]["user_id"] == "u1"


def test_profile_store_requires_user_id(tmp_path):
    from learning_agent.rag.store import ProfileStore

    with pytest.raises(ValueError):
        ProfileStore(tmp_path / "db2", embed_fn=fake_embed).save({"subject": "x"})


def test_resource_store_filters_by_run(tmp_path):
    pytest.importorskip("llama_index.vector_stores.chroma")
    from llama_index.core.embeddings import BaseEmbedding

    from learning_agent.rag.store import ResourceStore

    class Fake(BaseEmbedding):
        def _get_text_embedding(self, t): return fake_embed(t)
        def _get_query_embedding(self, q): return fake_embed(q)
        async def _aget_query_embedding(self, q): return fake_embed(q)

    store = ResourceStore(tmp_path / "db3", embed_model=Fake())
    store.add([{"title": "Embeddings", "url": "https://a", "description": "d", "difficulty": None}], run_id="r1")
    store.add([{"title": "Other run", "url": "https://b", "description": "d"}], run_id="r2")
    hits = store.retrieve("Embeddings", run_id="r1", k=5)
    assert [h["url"] for h in hits] == ["https://a"]
