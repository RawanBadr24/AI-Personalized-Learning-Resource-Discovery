import csv

from learning_agent.config import config
from learning_agent.tools import search
from learning_agent.tools.search import (
    dedupe_resources, normalize_url, parse_github, parse_huggingface,
    parse_web, parse_youtube, result_save, search_all_platforms,
)


def test_parse_github():
    payload = {"items": [{"full_name": "run-llama/llama_index", "html_url": "https://github.com/run-llama/llama_index",
                          "description": "RAG framework", "stargazers_count": 40000}]}
    [r] = parse_github(payload)
    assert r["platform"] == "GitHub" and r["price"] == "Free" and "40000" in r["description"]


def test_parse_huggingface_skips_items_without_id():
    out = parse_huggingface([{"id": "Qwen/Qwen3-8B", "downloads": 5, "pipeline_tag": "text-generation"}, {}])
    assert len(out) == 1 and out[0]["url"] == "https://huggingface.co/Qwen/Qwen3-8B"


def test_parse_youtube():
    payload = {"items": [{"id": {"videoId": "abc"}, "snippet": {"title": "T", "channelTitle": "C", "description": "D"}},
                         {"id": {}, "snippet": {}}]}
    [r] = parse_youtube(payload)
    assert r["url"].endswith("abc") and r["price"] == "Free"


def test_parse_web_price_is_unknown():
    [r] = parse_web([{"title": "T", "href": "https://x.com/a", "body": "b"}, {"title": "no url"}])
    assert r["price"] == "Unknown"


def test_normalize_url_and_dedupe():
    assert normalize_url("https://www.Example.com/A/") == normalize_url("http://example.com/a")
    items = [{"url": "https://a.com/x"}, {"url": "https://www.a.com/x/"}, {"url": "https://b.com"}]
    assert len(dedupe_resources(items)) == 2
    assert len(dedupe_resources(items, seen={normalize_url("https://a.com/x")})) == 1


def test_search_all_platforms_skips_youtube_without_key_and_survives_failures(monkeypatch):
    monkeypatch.setattr(config, "youtube_api_key", "")
    called = []

    def ok(q, n):
        called.append("github")
        return [{"url": "https://g.com/1", "title": "g"}]

    def boom(q, n):
        raise RuntimeError("down")

    def yt(q, n):  # must never be called
        called.append("youtube")
        return []

    monkeypatch.setitem(search.SEARCHERS, "github", ok)
    monkeypatch.setitem(search.SEARCHERS, "web", boom)
    monkeypatch.setitem(search.SEARCHERS, "youtube", yt)
    out = search_all_platforms("q", ["github", "web", "youtube", "nonsense"], 3)
    assert [r["url"] for r in out] == ["https://g.com/1"] and out[0]["query"] == "q"
    assert called == ["github"]


def test_youtube_used_when_key_present(monkeypatch):
    monkeypatch.setattr(config, "youtube_api_key", "k")
    monkeypatch.setitem(search.SEARCHERS, "youtube", lambda q, n: [{"url": "https://y.com/1"}])
    assert len(search_all_platforms("q", ["youtube"], 1)) == 1


def test_result_save_merges_and_dedupes(tmp_path):
    path = tmp_path / "r.csv"
    result_save([{"title": "A", "url": "https://a.com", "platform": "Web"}], path)
    result_save([{"title": "A2", "url": "https://a.com/", "platform": "Web"}, {"title": "B", "url": "https://b.com"}], path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["title"] for r in rows] == ["A", "B"]


def test_available_platforms_hides_youtube_without_key(monkeypatch):
    monkeypatch.setattr(config, "youtube_api_key", "")
    assert "youtube" not in search.available_platforms()
    monkeypatch.setattr(config, "youtube_api_key", "k")
    assert "youtube" in search.available_platforms()
