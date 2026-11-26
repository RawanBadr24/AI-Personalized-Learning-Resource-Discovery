"""Shared fixtures. No network, no real LLM, no API keys needed."""
import json

import pytest


class FakeLLM:
    """Replaces `learning_agent.utils.llm_complete`. Set `.handler` to a function(prompt) -> dict|list|str."""

    def __init__(self):
        self.handler = lambda prompt: {}
        self.calls = []

    def __call__(self, prompt, system=None):
        self.calls.append(prompt)
        out = self.handler(prompt)
        return out if isinstance(out, str) else json.dumps(out)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("learning_agent.utils.llm_complete", fake)
    return fake


@pytest.fixture
def profile():
    return {
        "user_id": "u1",
        "subject": "RAG with LlamaIndex",
        "objectives": ["build a RAG app"],
        "level": "beginner",
        "topics": ["Embeddings", "Vector stores", "Query engines"],
        "knowledge_gaps": ["vector databases"],
        "preferences": {"budget": "free", "platforms": [], "language": "English", "resource_types": []},
        "hours_per_week": 5,
    }


def make_report(url, passed=True, **over):
    base = dict(url=url, title=url, relevance=0.9, coverage=0.8, level_match=0.8, quality=0.7, cost=1.0,
                score=0.85, passed=passed, topics_covered=[], issues=[], reason="ok")
    base.update(over)
    return base


class FakeStore:
    def __init__(self):
        self.d = {}

    async def get(self, key, default=None):
        return self.d.get(key, default)

    async def set(self, key, value):
        self.d[key] = value


class FakeCtx:
    """Minimal stand-in for the LlamaIndex workflow Context (`ctx.store.get/set`)."""

    def __init__(self):
        self.store = FakeStore()


@pytest.fixture
def ctx(profile):
    from learning_agent.agents.state import STATE_KEY, new_state

    c = FakeCtx()
    c.store.d[STATE_KEY] = new_state("u1")
    return c


@pytest.fixture
def ctx_with_profile(ctx, profile):
    from learning_agent.agents.state import STATE_KEY

    ctx.store.d[STATE_KEY]["profile"] = profile
    return ctx
