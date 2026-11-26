"""Tests for the state-aware agent tools (no LLM, no network, no llama-index needed)."""
import csv

import pytest

from learning_agent.agents import tools as T
from learning_agent.agents.state import STATE_KEY, get_state
from learning_agent.config import config
from learning_agent.tools import profile as ptools


class FakeIndex:
    def __init__(self):
        self.added = []

    def add(self, resources, run_id):
        self.added += resources

    def retrieve(self, query, run_id, k=5):
        return [dict(r, score=0.9) for r in self.added[:k]]


def results(prefix, n=3):
    return [{"title": f"{prefix} {i}", "url": f"https://{prefix}.com/{i}", "platform": "Web", "description": "d",
             "price": "Free", "difficulty": "", "language": "", "duration": "", "query": "q"} for i in range(n)]


@pytest.fixture
def env(monkeypatch, tmp_path, ctx_with_profile):
    index = FakeIndex()
    monkeypatch.setattr(T, "_resource_store", lambda: index)
    monkeypatch.setattr(config, "data_dir", tmp_path)
    monkeypatch.setattr(config, "youtube_api_key", "")
    monkeypatch.setattr(T, "search_all_platforms", lambda q, platforms, n: results(q.replace(" ", ""), 3))
    return ctx_with_profile


def ev(rid, score=0.9, topics=("Embeddings",)):
    return {"id": rid, "relevance": score, "coverage": score, "level_match": score, "quality": score,
            "topics_covered": list(topics), "reason": "r"}


# ------------------------------------------------------------------ InfoAgent
async def test_save_profile_requires_core_fields(ctx, monkeypatch):
    monkeypatch.setattr(ptools, "_store", lambda: None)  # must not be reached
    out = await T.save_profile(ctx, {"subject": "RAG"})
    assert out.startswith("FAILURE") and (await get_state(ctx))["profile"] is None


async def test_save_profile_stores_in_chroma_and_state(ctx, monkeypatch):
    saved = {}

    class S:
        def save(self, p): saved.update(p); return p
        def get(self, uid): return saved

    monkeypatch.setattr(ptools, "_store", lambda: S())
    out = await T.save_profile(ctx, {"user_id": "someone-else", "subject": "RAG", "level": "beginner", "topics": "A, B"})
    assert out.startswith("SUCCESS") and "SearchAgent" in out
    state = await get_state(ctx)
    assert state["profile"]["user_id"] == "u1"            # the agent cannot overwrite the real user id
    assert state["profile"]["topics"] == ["A", "B"]


# ------------------------------------------------------------------ SearchAgent
async def test_search_requires_profile(ctx):
    assert (await T.search_resources(ctx, "q")).startswith("ERROR")


async def test_search_stores_dedupes_and_assigns_ids(env, tmp_path):
    out = await T.search_resources(env, "embeddings tutorial")
    assert "3 new" in out
    again = await T.search_resources(env, "embeddings tutorial")        # same urls -> all duplicates
    assert "0 new" in again
    state = await get_state(env)
    assert list(state["resources"]) == ["R1", "R2", "R3"] and state["pending"] == ["R1", "R2", "R3"]
    rows = list(csv.DictReader((tmp_path / "resources.csv").open(encoding="utf-8")))
    assert len(rows) == 3


async def test_search_budget_per_cycle(env):
    for i in range(config.max_searches_per_cycle):
        assert "BUDGET" not in await T.search_resources(env, f"query {i}")
    assert "SEARCH BUDGET USED" in await T.search_resources(env, "one too many")


async def test_search_reports_skipped_platform(env):
    out = await T.search_resources(env, "q", platforms=["youtube", "web"])
    assert "Skipped unavailable platforms: ['youtube']" in out


# ------------------------------------------------------------------ EvaluationAgent
async def test_pending_then_record_moves_resources(env):
    await T.search_resources(env, "q")
    pending = await T.get_pending_resources(env, limit=2)
    assert "[R1]" in pending and "[R2]" in pending and "[R3]" not in pending
    out = await T.record_evaluations(env, [ev("R1"), ev("R2", 0.1), {**ev("R1"), "id": "R99"}, {"bad": "x"}])
    assert "[R1] score" in out and "PASS" in out and "FAIL" in out
    assert "unknown resource id" in out and "REJECTED" in out
    state = await get_state(env)
    assert state["pending"] == ["R3"] and state["reports"]["R1"]["passed"] and not state["reports"]["R2"]["passed"]


async def test_evaluation_budget(env, monkeypatch):
    monkeypatch.setattr(config, "max_resources_to_evaluate", 1)
    await T.search_resources(env, "q")
    await T.record_evaluations(env, [ev("R1")])
    assert "EVALUATION BUDGET USED" in await T.get_pending_resources(env)


async def test_coverage_advice_and_cycle_limit(env):
    await T.search_resources(env, "q")
    await T.record_evaluations(env, [ev("R1"), ev("R2"), ev("R3")])
    assert "request_more_search" in await T.check_topic_coverage(env)       # Vector stores/Query engines missing

    assert (await T.request_more_search(env, ["Vector stores"], ["none found"], ["vector store basics"])).startswith("OK")
    assert (await T.request_more_search(env, ["Vector stores"], ["none"], ["x"])).startswith("OK")   # cycle 3
    refused = await T.request_more_search(env, ["Vector stores"], ["none"], ["y"])
    assert "CYCLE LIMIT REACHED" in refused
    state = await get_state(env)
    assert state["cycle"] == config.max_cycles == 3
    assert state["feedback"]["suggested_queries"] == ["x"]
    assert "finalize_evaluation" in await T.check_topic_coverage(env)


async def test_new_cycle_resets_budgets(env):
    for i in range(config.max_searches_per_cycle):
        await T.search_resources(env, f"query {i}")
    await T.request_more_search(env, ["x"], ["r"], ["new"])
    assert "BUDGET" not in await T.search_resources(env, "fresh query")


async def test_finalize_refuses_when_nothing_passed_and_cycles_remain(env):
    await T.search_resources(env, "q")
    await T.record_evaluations(env, [ev("R1", 0.1)])
    assert "NOT FINALIZED" in await T.finalize_evaluation(env)


async def test_finalize_ranks_only_passed_resources(env, tmp_path):
    await T.search_resources(env, "q")
    await T.record_evaluations(env, [ev("R1", 0.95), ev("R2", 0.75, ["Vector stores"]), ev("R3", 0.1)])
    out = await T.finalize_evaluation(env)
    state = await get_state(env)
    assert "SUCCESS" in out and [r["id"] for r in state["ranked"]] == ["R1", "R2"]
    assert state["grouped"]["Embeddings"][0]["id"] == "R1" and state["grouped"]["Query engines"] == []
    assert "Query engines" in out                                   # gap is reported
    assert (tmp_path / "ranked_resources.csv").exists()


async def test_finalize_at_cycle_limit_works_with_nothing_passed(env):
    state = await get_state(env)
    state["cycle"] = config.max_cycles
    await T.put_state(env, state)
    out = await T.finalize_evaluation(env)
    assert "Ranked 0 resources" in out


async def test_find_in_index_shows_status(env):
    await T.search_resources(env, "q")
    await T.record_evaluations(env, [ev("R1")])
    out = await T.find_in_index(env, "embeddings")
    assert "[R1]" in out and "PASSED" in out and "not evaluated" in out


# ------------------------------------------------------------------ LearningPathAgent
async def ranked_env(env):
    await T.search_resources(env, "q")
    await T.record_evaluations(env, [ev("R1", 0.95), ev("R2", 0.85, ["Vector stores"]), ev("R3", 0.8, ["Query engines"])])
    await T.finalize_evaluation(env)
    return env


async def test_save_roadmap_orders_validates_and_blocks_invented_ids(env):
    await ranked_env(env)
    listing = await T.get_ranked_resources(env)
    assert "TOPIC: Embeddings" in listing and "[R1]" in listing
    out = await T.save_roadmap(
        env,
        prerequisites={"Query engines": ["Vector stores"], "Vector stores": ["Embeddings"]},
        resources_by_topic={"embeddings": ["R1", "R42"], "Vector stores": ["R2"], "Query engines": ["R3"]},
        practice_tasks_by_topic={"Embeddings": ["embed a sentence"], "Vector stores": ["store 10 docs"], "Query engines": ["ask a question"]},
    )
    assert out.startswith("SUCCESS") and "R42" in out                    # invented id flagged
    road = (await get_state(env))["roadmap"]
    assert [s["topic"] for s in road["stages"]] == ["Embeddings", "Vector stores", "Query engines"]
    assert [r["id"] for r in road["stages"][0]["resources"]] == ["R1"]   # R42 dropped
    assert road["stages"][0]["practice_tasks"] == ["embed a sentence"]


async def test_save_roadmap_falls_back_to_grouped_resources(env):
    await ranked_env(env)
    out = await T.save_roadmap(env, {}, {}, {})
    assert "no resources chosen" in out and "no practice tasks" in out
    road = (await get_state(env))["roadmap"]
    assert road["stages"][0]["resources"][0]["id"] == "R1"


async def test_brief_reports_progress(env):
    import json

    await T.search_resources(env, "q")
    brief = json.loads(await T.get_learning_brief(env))
    assert brief["cycle"] == "1/3" and brief["counts"]["pending_evaluation"] == 3 and brief["queries_already_used"] == ["q"]
