"""Agent 2 tools: multi-platform search + CSV persistence.

Free/easy sources: GitHub, Hugging Face Hub, DuckDuckGo (web). YouTube is optional and is skipped
automatically when YOUTUBE_API_KEY is not set.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from urllib.parse import urlsplit

import requests

from learning_agent.config import PLATFORMS, config

log = logging.getLogger(__name__)

FIELDS = ["title", "url", "platform", "description", "difficulty", "language", "duration", "price", "query"]


def make_resource(title, url, platform, description="", price="Unknown", **extra) -> dict:
    return {
        "title": (title or "").strip(),
        "url": (url or "").strip(),
        "platform": platform,
        "description": (description or "").strip()[:500],
        "difficulty": extra.get("difficulty", ""),
        "language": extra.get("language", ""),
        "duration": extra.get("duration", ""),
        "price": price,
        "query": extra.get("query", ""),
    }


# ----------------------------------------------------------------------------- HTTP / parsers
def _http_get(url: str, params: dict | None = None, headers: dict | None = None):
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_github(payload: dict) -> list[dict]:
    out = []
    for item in payload.get("items", []):
        desc = item.get("description") or ""
        stars = item.get("stargazers_count")
        if stars is not None:
            desc = f"{desc} (★ {stars})".strip()
        out.append(make_resource(item.get("full_name"), item.get("html_url"), "GitHub", desc, price="Free"))
    return out


def parse_huggingface(payload: list) -> list[dict]:
    out = []
    for item in payload or []:
        model_id = item.get("id") or item.get("modelId")
        if not model_id:
            continue
        desc = f"{item.get('pipeline_tag') or 'model'} · {item.get('downloads', 0)} downloads"
        out.append(make_resource(model_id, f"https://huggingface.co/{model_id}", "Hugging Face", desc, price="Free"))
    return out


def parse_youtube(payload: dict) -> list[dict]:
    out = []
    for item in payload.get("items", []):
        vid = (item.get("id") or {}).get("videoId")
        snip = item.get("snippet") or {}
        if vid:
            desc = f"{snip.get('channelTitle', '')}: {snip.get('description', '')}"
            out.append(make_resource(snip.get("title"), f"https://www.youtube.com/watch?v={vid}", "YouTube", desc, price="Free"))
    return out


def parse_web(results: list[dict]) -> list[dict]:
    # Web pages: price cannot be known from a search snippet, so we stay honest and say Unknown.
    return [
        make_resource(r.get("title"), r.get("href") or r.get("url"), "Web", r.get("body", ""), price="Unknown")
        for r in results or []
        if (r.get("href") or r.get("url"))
    ]


# ----------------------------------------------------------------------------- platform searchers
def _search_github(query: str, n: int) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if config.github_token:
        headers["Authorization"] = f"Bearer {config.github_token}"
    payload = _http_get(
        "https://api.github.com/search/repositories",
        {"q": query, "sort": "stars", "order": "desc", "per_page": n},
        headers,
    )
    return parse_github(payload)


def _search_huggingface(query: str, n: int) -> list[dict]:
    payload = _http_get(
        "https://huggingface.co/api/models",
        {"search": query, "limit": n, "sort": "downloads", "direction": -1},
    )
    return parse_huggingface(payload)


def _ddg_text(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    return list(DDGS().text(query, max_results=n))


def _search_web(query: str, n: int) -> list[dict]:
    return parse_web(_ddg_text(query, n))


def _search_youtube(query: str, n: int) -> list[dict]:
    payload = _http_get(
        "https://www.googleapis.com/youtube/v3/search",
        {"part": "snippet", "type": "video", "q": query, "maxResults": n, "key": config.youtube_api_key},
    )
    return parse_youtube(payload)


SEARCHERS = {
    "github": _search_github,
    "huggingface": _search_huggingface,
    "web": _search_web,
    "youtube": _search_youtube,
}


def available_platforms() -> list[str]:
    """Platforms that can actually be searched right now (YouTube needs an API key)."""
    return [p for p in SEARCHERS if p != "youtube" or config.youtube_api_key]


def search_all_platforms(
    query: str, platforms: list[str] | None = None, max_results_per_platform: int = 5
) -> list[dict]:
    """Search several free educational platforms (GitHub, Hugging Face, web, optional YouTube) and return results in one unified format.

    Unknown platforms are ignored; YouTube is skipped when no API key is configured. A failure on one platform never stops the others.
    """
    results: list[dict] = []
    for name in platforms or list(config.default_platforms):
        name = name.lower().strip()
        if name not in SEARCHERS:
            log.info("Skipping unsupported platform %r (supported: %s)", name, PLATFORMS)
            continue
        if name == "youtube" and not config.youtube_api_key:
            log.info("YouTube skipped: YOUTUBE_API_KEY not set")
            continue
        try:
            for r in SEARCHERS[name](query, max_results_per_platform):
                r["query"] = query
                results.append(r)
        except Exception as exc:  # noqa: BLE001
            log.warning("Search on %s failed: %s", name, exc)
    return results


# ----------------------------------------------------------------------------- dedupe / save
def normalize_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    return f"{parts.netloc.lower().removeprefix('www.')}{parts.path.rstrip('/')}".lower()


def dedupe_resources(resources: list[dict], seen: set[str] | None = None) -> list[dict]:
    """Drop duplicates (by normalized URL), also against URLs in `seen`."""
    seen = set(seen or ())
    out = []
    for r in resources:
        key = normalize_url(r.get("url", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def result_save(resources: list[dict], path: str | Path | None = None) -> str:
    """Save collected resources to a CSV file (merged with existing rows, deduplicated by URL). Returns the file path."""
    path = Path(path or config.resources_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    merged = dedupe_resources(existing + list(resources))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    return str(path)
