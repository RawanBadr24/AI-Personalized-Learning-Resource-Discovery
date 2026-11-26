"""ChromaDB-backed stores. One persistent DB, two collections:

* `user_profiles`      -> ProfileStore  (exact lookup by user_id + semantic recall)
* `learning_resources` -> ResourceStore (LlamaIndex VectorStoreIndex; topic retrieval per run)
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

from learning_agent.config import config


@lru_cache(maxsize=4)
def get_chroma_client(path: str):
    import chromadb

    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def profile_to_text(profile: dict) -> str:
    prefs = profile.get("preferences") or {}
    parts = [
        f"Subject: {profile.get('subject', '')}",
        f"Level: {profile.get('level', '')}",
        f"Objectives: {'; '.join(profile.get('objectives') or [])}",
        f"Topics: {'; '.join(profile.get('topics') or [])}",
        f"Preferences: {json.dumps(prefs)}",
    ]
    return "\n".join(parts)


class ProfileStore:
    """User profiles. `embed_fn` maps text -> vector (defaults to the local HF embedding model)."""

    def __init__(self, persist_dir: str | Path | None = None, embed_fn: Callable | None = None):
        client = get_chroma_client(str(persist_dir or config.chroma_dir))
        self._col = client.get_or_create_collection("user_profiles", metadata={"hnsw:space": "cosine"})
        self._embed_fn = embed_fn

    def _embed(self, text: str):
        if self._embed_fn is None:
            from learning_agent.llm import get_embed_model

            self._embed_fn = get_embed_model().get_text_embedding
        return self._embed_fn(text)

    def get(self, user_id: str) -> dict:
        res = self._col.get(ids=[user_id], include=["metadatas"])
        if not res["ids"]:
            return {}
        return json.loads(res["metadatas"][0]["profile_json"])

    def save(self, profile: dict) -> dict:
        user_id = str(profile.get("user_id") or "").strip()
        if not user_id:
            raise ValueError("profile must contain a non-empty 'user_id'")
        profile = {**profile, "user_id": user_id, "saved_at": time.time()}
        text = profile_to_text(profile)
        self._col.upsert(
            ids=[user_id],
            documents=[text],
            embeddings=[self._embed(text)],
            metadatas=[{"profile_json": json.dumps(profile), "subject": str(profile.get("subject", ""))}],
        )
        return profile

    def similar(self, query: str, k: int = 3) -> list[dict]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[self._embed(query)], n_results=min(k, self._col.count()), include=["metadatas"]
        )
        return [json.loads(m["profile_json"]) for m in res["metadatas"][0]]


def _clean_meta(resource: dict, run_id: str) -> dict:
    meta = {k: ("" if v is None else v) for k, v in resource.items() if isinstance(v, (str, int, float, bool, type(None)))}
    meta["run_id"] = run_id
    return meta


class ResourceStore:
    """Searched resources, embedded so agents can retrieve them by topic (RAG)."""

    def __init__(self, persist_dir: str | Path | None = None, embed_model=None):
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        client = get_chroma_client(str(persist_dir or config.chroma_dir))
        collection = client.get_or_create_collection("learning_resources")
        if embed_model is None:
            from learning_agent.llm import get_embed_model

            embed_model = get_embed_model()
        self.index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=collection), embed_model=embed_model
        )

    def add(self, resources: list[dict], run_id: str) -> int:
        from llama_index.core.schema import TextNode

        nodes = []
        for r in resources:
            node_id = hashlib.md5(f"{run_id}|{r.get('url', '')}".encode()).hexdigest()
            text = f"{r.get('title', '')}\n{r.get('description', '')}"
            nodes.append(TextNode(id_=node_id, text=text, metadata=_clean_meta(r, run_id)))
        if nodes:
            self.index.insert_nodes(nodes)
        return len(nodes)

    def retrieve(self, query: str, run_id: str, k: int = 5) -> list[dict]:
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

        retriever = self.index.as_retriever(
            similarity_top_k=k,
            filters=MetadataFilters(filters=[MetadataFilter(key="run_id", value=run_id)]),
        )
        return [{**n.node.metadata, "score": n.score} for n in retriever.retrieve(query)]


@lru_cache(maxsize=1)
def get_profile_store() -> ProfileStore:
    return ProfileStore()


@lru_cache(maxsize=1)
def get_resource_store() -> ResourceStore:
    return ResourceStore()
