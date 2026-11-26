"""Agent 5 tools: prerequisites, ordering, roadmap, practice tasks."""
from __future__ import annotations

import re

from learning_agent.config import config


def sanitize_prerequisites(topics: list[str], prerequisites: dict | None) -> dict:
    """Clean an agent-written prerequisite map: canonical topic names only, no self/unknown references."""
    lookup = {t.lower(): t for t in topics}
    result: dict[str, list[str]] = {t: [] for t in topics}
    for topic, prereqs in (prerequisites or {}).items():
        canon = lookup.get(str(topic).strip().lower())
        if not canon:
            continue
        for p in prereqs or []:
            pc = lookup.get(str(p).strip().lower())
            if pc and pc != canon and pc not in result[canon]:
                result[canon].append(pc)
    return result


def build_topic_dependencies(topics: list[str], prerequisites: dict) -> list[str]:
    """Order topics so prerequisites come first (stable topological sort; cycles fall back to the original order)."""
    remaining = list(topics)
    ordered: list[str] = []
    while remaining:
        ready = [t for t in remaining if all(p in ordered for p in prerequisites.get(t, []) if p in topics)]
        if not ready:  # cycle -> keep original order for what's left
            ordered.extend(remaining)
            break
        ordered.append(ready[0])
        remaining.remove(ready[0])
    return ordered


def parse_duration_hours(text: str) -> float | None:
    """Parse '1h 30m', '45 min', '2 hours', 'PT1H30M' -> hours. Returns None if unparseable."""
    if not text:
        return None
    t = text.lower()
    h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", t) or re.search(r"pt(\d+)h", t)
    m = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", t) or re.search(r"(\d+)m\b", t)
    if not h and not m:
        return None
    return round((float(h.group(1)) if h else 0) + (float(m.group(1)) if m else 0) / 60, 2)


def generate_roadmap(user_profile: dict, stages: list[dict]) -> dict:
    """Assemble an ordered roadmap with objectives, resources, estimated time and practice tasks.

    Each stage dict: {"topic": str, "resources": [ranked resources], "practice_tasks": [str]}.
    Time comes from resource durations when known; otherwise a default estimate (clearly labelled).
    """
    out_stages = []
    total = 0.0
    for i, st in enumerate(stages, 1):
        resources = st.get("resources", [])
        durations = [d for d in (parse_duration_hours(r.get("duration", "")) for r in resources) if d]
        hours, basis = (sum(durations), "from resource durations") if durations else (config.default_hours_per_topic, "default estimate")
        total += hours
        out_stages.append(
            {
                "stage": i,
                "topic": st["topic"],
                "objective": f"Understand and apply: {st['topic']}",
                "resources": resources,
                "estimated_hours": round(hours, 1),
                "time_basis": basis,
                "practice_tasks": st.get("practice_tasks", []),
                "note": "" if resources else "No approved resource found for this topic.",
            }
        )
    weekly = user_profile.get("hours_per_week")
    return {
        "subject": user_profile.get("subject", ""),
        "level": user_profile.get("level", ""),
        "stages": out_stages,
        "total_estimated_hours": round(total, 1),
        "estimated_weeks": round(total / weekly, 1) if isinstance(weekly, (int, float)) and weekly > 0 else None,
    }
