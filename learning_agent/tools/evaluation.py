"""Deterministic evaluation helpers. The EvaluationAgent does the *judging*; this code does the math."""
from __future__ import annotations

from learning_agent.config import config
from learning_agent.utils import clamp01


def cost_score(price: str) -> float:
    """Rule-based cost score: free=1.0, unknown=0.5, paid=0.0."""
    p = (price or "").strip().lower()
    if "free" in p or p in {"0", "$0"}:
        return 1.0
    if p in {"", "unknown", "n/a"}:
        return 0.5
    return 0.0


def is_valid_url(url: str) -> bool:
    return (url or "").startswith(("http://", "https://"))


def calculate_evaluation_score(scores: dict, weights: dict) -> float:
    """Weighted average of the criteria present in both `scores` and `weights` (weights are normalised)."""
    used = {k: w for k, w in weights.items() if k in scores}
    total = sum(used.values())
    if total <= 0:
        return 0.0
    return round(sum(scores[k] * w for k, w in used.items()) / total, 4)


def score_evaluation(judgment: dict, resource: dict, requested_topics: list[str]) -> dict:
    """Turn the agent's judgment (relevance/coverage/level_match/quality/topics_covered/...) into a full report.

    Clamps scores to [0,1], canonicalises topic names, adds the rule-based cost score, the weighted
    score and the pass/fail decision. An invalid URL forces relevance to 0.
    """
    lookup = {t.lower(): t for t in requested_topics}
    covered = []
    for t in judgment.get("topics_covered") or []:
        canon = lookup.get(str(t).strip().lower())
        if canon and canon not in covered:
            covered.append(canon)
    report = {
        "relevance": clamp01(judgment.get("relevance")),
        "coverage": clamp01(judgment.get("coverage")),
        "level_match": clamp01(judgment.get("level_match")),
        "quality": clamp01(judgment.get("quality")),
        "topics_covered": covered,
        "issues": list(judgment.get("issues") or []),
        "reason": judgment.get("reason", "") or "",
    }
    if not is_valid_url(resource.get("url", "")):
        report["relevance"] = 0.0
        report["issues"].append("invalid or missing URL")
    report["cost"] = cost_score(resource.get("price", ""))
    report["score"] = calculate_evaluation_score(report, config.eval_weights)
    report["passed"] = report["score"] >= config.pass_threshold and report["relevance"] > 0.3
    report["url"] = resource.get("url", "")
    report["title"] = resource.get("title", "")
    report["id"] = resource.get("id", "")
    return report


def topic_coverage(approved_reports: list[dict], topics: list[str], min_per_topic: int = 1) -> tuple[list[str], list[str]]:
    """Return (covered_topics, missing_topics) given the reports of approved resources."""
    counts = {t: 0 for t in topics}
    for rep in approved_reports:
        for t in rep.get("topics_covered", []):
            if t in counts:
                counts[t] += 1
    covered = [t for t in topics if counts[t] >= min_per_topic]
    return covered, [t for t in topics if t not in covered]
