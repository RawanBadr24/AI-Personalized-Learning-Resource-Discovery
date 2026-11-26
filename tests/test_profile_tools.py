import pytest

from learning_agent.tools import profile as ptools


class FakeStore:
    def __init__(self):
        self.data = {}

    def get(self, uid):
        return self.data.get(uid, {})

    def save(self, p):
        if not p.get("user_id"):
            raise ValueError("no user_id")
        self.data[p["user_id"]] = p
        return p


@pytest.fixture
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr(ptools, "_store", lambda: s)
    return s


def test_data_save_and_retrieve(store):
    assert ptools.profile_retrieve("u1") == {}
    assert ptools.data_save({"user_id": "u1", "subject": "x"}).startswith("SUCCESS")
    assert ptools.profile_retrieve("u1")["subject"] == "x"


def test_data_save_reports_failure(store):
    assert ptools.data_save({"subject": "x"}).startswith("FAILURE")


def test_analyze_goal_fills_missing_keys(fake_llm):
    fake_llm.handler = lambda p: {"subject": "RAG"}
    out = ptools.analyze_learning_goal("learn rag")
    assert out == {"subject": "RAG", "objectives": [], "expected_outcomes": [], "requested_topics": []}


def test_assess_defaults_to_free_and_english(fake_llm):
    fake_llm.handler = lambda p: {}
    out = ptools.assess_user_level_and_preferences("RAG", "beginner", [])
    assert out["level"] == "beginner" and out["preferences"]["budget"] == "free" and out["preferences"]["language"] == "English"


def test_goal_decomposer(fake_llm):
    fake_llm.handler = lambda p: ["A", "B"]
    assert ptools.goal_decomposer("s", "o", "beginner") == ["A", "B"]
    fake_llm.handler = lambda p: {"not": "a list"}
    assert ptools.goal_decomposer("s", "o", "beginner") == []
