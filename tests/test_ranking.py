import csv

from learning_agent.tools.ranking import group_resources_by_topic, rank_resources, ranking_save
from tests.conftest import make_report


def res(url, **kw):
    return {"title": url, "url": url, "platform": "Web", "price": "Free", **kw}


def test_failed_and_unreported_resources_are_excluded(profile):
    resources = [res("a"), res("b"), res("c")]
    reports = [make_report("a"), make_report("b", passed=False)]
    ranked = rank_resources(resources, reports, profile)
    assert [r["url"] for r in ranked] == ["a"]


def test_ranking_order_and_rank_numbers(profile):
    resources = [res("low"), res("high")]
    reports = [make_report("low", relevance=0.5, level_match=0.5), make_report("high", relevance=1, level_match=1)]
    ranked = rank_resources(resources, reports, profile)
    assert [r["url"] for r in ranked] == ["high", "low"] and [r["rank"] for r in ranked] == [1, 2]


def test_free_preference_boosts_free_resources(profile):
    resources = [res("free"), res("paid", price="$99")]
    reports = [make_report("free", cost=1.0, relevance=0.8), make_report("paid", cost=0.0, relevance=0.85)]
    assert rank_resources(resources, reports, profile)[0]["url"] == "free"


def test_group_by_topic_uses_labels_then_keyword_fallback():
    ranked = [res("a", topics_covered=["Embeddings"]), res("b", title="intro to vector stores", description="")]
    groups = group_resources_by_topic(ranked, ["Embeddings", "Vector stores", "Query engines"])
    assert [r["url"] for r in groups["Embeddings"]] == ["a"]
    assert [r["url"] for r in groups["Vector stores"]] == ["b"]
    assert groups["Query engines"] == []


def test_ranking_save(tmp_path):
    path = tmp_path / "rank.csv"
    msg = ranking_save([{"rank": 1, "final_score": 0.9, "title": "A", "url": "u", "topics_covered": ["X", "Y"]}], path)
    assert msg.startswith("SUCCESS")
    assert list(csv.DictReader(path.open()))[0]["topics_covered"] == "X; Y"
