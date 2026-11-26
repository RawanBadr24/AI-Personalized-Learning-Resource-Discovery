"""Agent 4 tools: ranking, grouping, saving."""
from __future__ import annotations

import csv
from pathlib import Path

from learning_agent.config import config
from learning_agent.tools.evaluation import calculate_evaluation_score


def rank_resources(
    resources: list[dict], evaluation_reports: list[dict], user_profile: dict, weights: dict | None = None
) -> list[dict]:
    """Rank approved resources by weighted relevance, level match, cost, quality and practical value (best first).

    Resources whose evaluation report is missing or failed are excluded.
    """
    weights = dict(weights or config.ranking_weights)
    budget = str((user_profile.get("preferences") or {}).get("budget", "free")).lower()
    if "free" in budget:  # user wants free -> make cost matter more
        weights["cost"] = weights.get("cost", 0.2) * 1.5
    reports = {r["url"]: r for r in evaluation_reports}
    ranked = []
    for res in resources:
        rep = reports.get(res.get("url"))
        if not rep or not rep.get("passed"):
            continue
        scores = {
            "relevance": rep["relevance"],
            "level_match": rep["level_match"],
            "cost": rep["cost"],
            "quality": rep["quality"],
            "practical": rep["coverage"],
        }
        ranked.append(
            {
                **res,
                "final_score": calculate_evaluation_score(scores, weights),
                "topics_covered": rep.get("topics_covered", []),
                "reason": rep.get("reason", ""),
            }
        )
    ranked.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def group_resources_by_topic(resources: list[dict], topics: list[str]) -> dict:
    """Group ranked resources by the topics they cover (keeps rank order). Topics with no resource map to []."""
    groups: dict[str, list[dict]] = {t: [] for t in topics}
    for r in resources:
        covered = r.get("topics_covered") or [
            t for t in topics if t.lower() in f"{r.get('title', '')} {r.get('description', '')}".lower()
        ]
        for t in covered:
            if t in groups:
                groups[t].append(r)
    return groups


def ranking_save(ranked_resources: list[dict], path: str | Path | None = None) -> str:
    """Save ranked resources to CSV. Returns a success or failure message."""
    path = Path(path or config.ranking_csv)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["rank", "final_score", "title", "url", "platform", "price", "difficulty", "topics_covered", "reason"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in ranked_resources:
                writer.writerow({**r, "topics_covered": "; ".join(r.get("topics_covered", []))})
        return f"SUCCESS: saved {len(ranked_resources)} ranked resources to {path}"
    except Exception as exc:  # noqa: BLE001
        return f"FAILURE: could not save ranking ({exc})"
