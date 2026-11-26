from learning_agent.config import config
from learning_agent.tools.evaluation import calculate_evaluation_score, cost_score, score_evaluation, topic_coverage
from tests.conftest import make_report

TOPICS = ["Embeddings", "Vector stores"]
RES = {"id": "R1", "title": "Embeddings 101", "url": "https://x.com/e", "platform": "Web", "price": "Free"}
GOOD = {"relevance": 0.9, "coverage": 0.9, "level_match": 0.9, "quality": 0.9, "topics_covered": ["Embeddings"]}


def test_cost_score():
    assert cost_score("Free") == 1.0 and cost_score("Unknown") == 0.5 and cost_score("$49") == 0.0 and cost_score("") == 0.5


def test_calculate_score_weighted_and_normalised():
    assert calculate_evaluation_score({"a": 1.0, "b": 0.0}, {"a": 0.75, "b": 0.25}) == 0.75
    assert calculate_evaluation_score({"a": 1.0, "b": 0.5}, {"a": 2, "b": 2}) == 0.75  # weights need not sum to 1
    assert calculate_evaluation_score({"x": 1}, {"a": 1}) == 0.0


def test_score_evaluation_clamps_and_canonicalises_topics():
    rep = score_evaluation({"relevance": 1.7, "coverage": 0.8, "level_match": "0.6", "quality": -1,
                            "topics_covered": ["embeddings", "EMBEDDINGS", "Not a topic"]}, RES, TOPICS)
    assert rep["relevance"] == 1.0 and rep["quality"] == 0.0 and rep["level_match"] == 0.6
    assert rep["topics_covered"] == ["Embeddings"] and rep["id"] == "R1"


def test_pass_and_fail():
    good = score_evaluation(GOOD, RES, TOPICS)
    assert good["passed"] and good["score"] >= config.pass_threshold and good["url"] == RES["url"]
    bad = score_evaluation({"relevance": 0.1, "coverage": 0.1, "level_match": 0.1, "quality": 0.1}, RES, TOPICS)
    assert not bad["passed"]


def test_low_relevance_fails_even_if_other_scores_are_high():
    rep = score_evaluation({**GOOD, "relevance": 0.2}, RES, TOPICS)
    assert not rep["passed"]


def test_invalid_url_zeroes_relevance():
    rep = score_evaluation(GOOD, {**RES, "url": "ftp://nope"}, TOPICS)
    assert rep["relevance"] == 0.0 and "invalid or missing URL" in rep["issues"] and not rep["passed"]


def test_paid_resource_scores_lower_than_free():
    free = score_evaluation(GOOD, RES, TOPICS)["score"]
    paid = score_evaluation(GOOD, {**RES, "price": "$99"}, TOPICS)["score"]
    assert paid < free


def test_missing_judgment_fields_do_not_crash():
    assert score_evaluation({}, RES, TOPICS)["passed"] is False


def test_topic_coverage():
    reports = [make_report("a", topics_covered=["Embeddings"]), make_report("b", topics_covered=["Embeddings"])]
    covered, missing = topic_coverage(reports, TOPICS, min_per_topic=1)
    assert covered == ["Embeddings"] and missing == ["Vector stores"]
    assert topic_coverage(reports, TOPICS, min_per_topic=3)[0] == []
