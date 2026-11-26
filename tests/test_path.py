from learning_agent.tools.path import (
    build_topic_dependencies, generate_roadmap, parse_duration_hours, sanitize_prerequisites,
)

TOPICS = ["Querying", "Embeddings", "Vector stores"]


def test_build_dependencies_orders_prerequisites_first():
    prereqs = {"Querying": ["Vector stores"], "Vector stores": ["Embeddings"], "Embeddings": []}
    assert build_topic_dependencies(TOPICS, prereqs) == ["Embeddings", "Vector stores", "Querying"]


def test_build_dependencies_is_stable_without_prereqs():
    assert build_topic_dependencies(TOPICS, {}) == TOPICS


def test_build_dependencies_survives_cycles():
    prereqs = {"Querying": ["Embeddings"], "Embeddings": ["Querying"]}
    assert sorted(build_topic_dependencies(TOPICS, prereqs)) == sorted(TOPICS)


def test_sanitize_prerequisites_cleans_agent_output():
    out = sanitize_prerequisites(TOPICS, {"querying": ["EMBEDDINGS", "Quantum physics", "Querying"], "Unknown": ["Embeddings"]})
    assert out["Querying"] == ["Embeddings"] and set(out) == set(TOPICS)
    assert sanitize_prerequisites(TOPICS, None) == {t: [] for t in TOPICS}


def test_parse_duration_hours():
    assert parse_duration_hours("1h 30m") == 1.5
    assert parse_duration_hours("45 min") == 0.75
    assert parse_duration_hours("2 hours") == 2.0
    assert parse_duration_hours("PT1H30M") == 1.5
    assert parse_duration_hours("") is None and parse_duration_hours("long") is None


def test_generate_roadmap_estimates_and_flags_gaps(profile):
    stages = [
        {"topic": "Embeddings", "resources": [{"title": "v", "url": "u", "duration": "2 hours"}], "practice_tasks": ["do it"]},
        {"topic": "Vector stores", "resources": [], "practice_tasks": []},
    ]
    road = generate_roadmap(profile, stages)
    s1, s2 = road["stages"]
    assert s1["estimated_hours"] == 2.0 and s1["time_basis"] == "from resource durations"
    assert s2["time_basis"] == "default estimate" and "No approved resource" in s2["note"]
    assert road["total_estimated_hours"] == 5.0 and road["estimated_weeks"] == 1.0  # 5 h / (5 h per week)
