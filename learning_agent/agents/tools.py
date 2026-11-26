"""State-aware tools used by the agents (each takes the workflow `ctx` as its first argument).

NOTE: no `from __future__ import annotations` here on purpose: LlamaIndex detects the `ctx: Context`
parameter from its real (non-string) annotation.
"""
import asyncio
import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

try:
    from llama_index.core.workflow import Context
except ImportError:  # lets the tools be imported and unit-tested without llama-index installed
    Context = object

from learning_agent.agents.state import get_state, locked, put_state
from learning_agent.config import config
from learning_agent.tools.evaluation import score_evaluation, topic_coverage
from learning_agent.tools.path import build_topic_dependencies, generate_roadmap, sanitize_prerequisites
from learning_agent.tools.profile import data_save, profile_retrieve
from learning_agent.tools.ranking import group_resources_by_topic, rank_resources, ranking_save
from learning_agent.tools.search import (
    available_platforms,
    dedupe_resources,
    normalize_url,
    result_save,
    search_all_platforms,
)

log = logging.getLogger(__name__)


def _resource_store():
    from learning_agent.rag.store import get_resource_store

    return get_resource_store()


def _topics(state: dict) -> list[str]:
    profile = state.get("profile") or {}
    return profile.get("topics") or [profile.get("subject", "")]


def _short(resource: dict, desc_chars: int = 220) -> str:
    return (
        f"[{resource['id']}] {resource.get('title', '')} | {resource.get('platform', '')} | "
        f"price: {resource.get('price') or 'Unknown'} | {resource.get('url', '')} | "
        f"{(resource.get('description') or '')[:desc_chars]}"
    )


# =============================================================== shared (all agents)
async def get_learning_brief(ctx: Context) -> str:
    """Read the saved user profile and the current progress: topics, level, preferences, cycle number, queries already used, evaluator feedback, available platforms and resource counts. Call this first."""
    state = await get_state(ctx)
    reports = state["reports"].values()
    brief = {
        "profile": state["profile"],
        "cycle": f"{state['cycle']}/{config.max_cycles}",
        "searches_this_cycle": f"{state['searches_this_cycle']}/{config.max_searches_per_cycle}",
        "queries_already_used": state["queries"],
        "evaluator_feedback": state["feedback"],
        "available_platforms": available_platforms(),
        "counts": {
            "found": len(state["resources"]),
            "pending_evaluation": len(state["pending"]),
            "evaluated": len(state["reports"]),
            "passed": sum(1 for r in reports if r["passed"]),
        },
    }
    return json.dumps(brief, ensure_ascii=False, default=str)


async def find_in_index(ctx: Context, query: str, k: int = 5) -> str:
    """Semantic (RAG) search over the resources collected in this run. Use it to check whether a topic is already covered, to find near-duplicates, or to look for a close match to a topic. Returns the best hits with their evaluation status."""
    state = await get_state(ctx)
    try:
        hits = await asyncio.to_thread(_resource_store().retrieve, query, state["run_id"], max(1, min(int(k), 10)))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: index search failed ({exc})"
    if not hits:
        return "No matching resources in the index."
    lines = []
    for h in hits:
        rep = state["reports"].get(h.get("id", ""))
        status = "not evaluated" if not rep else f"{'PASSED' if rep['passed'] else 'FAILED'} score {rep['score']:.2f}"
        lines.append(f"[{h.get('id', '?')}] {h.get('title', '')} | {h.get('url', '')} | similarity {h.get('score') or 0:.2f} | {status}")
    return "\n".join(lines)


# =============================================================== InfoAgent
async def save_profile(ctx: Context, user_profile: dict) -> str:
    """Save the user's learning profile and return a success or failure message. Required keys: subject, level, topics (list of short topic names). Also include objectives, expected_outcomes, known_topics, knowledge_gaps, preferences (budget, platforms, language, resource_types) and hours_per_week."""
    async with locked(ctx):
        state = await get_state(ctx)
        profile = {**user_profile, "user_id": state["user_id"]}
        if isinstance(profile.get("topics"), str):
            profile["topics"] = [t.strip() for t in profile["topics"].split(",") if t.strip()]
        missing = [k for k in ("subject", "level", "topics") if not profile.get(k)]
        if missing:
            return f"FAILURE: missing required fields {missing}. Ask the user for them before saving."
        message = await asyncio.to_thread(data_save, profile)
        if message.startswith("FAILURE"):
            return message
        saved = await asyncio.to_thread(profile_retrieve, state["user_id"])
        state["profile"] = saved or profile
        await put_state(ctx, state)
        return message + " Now hand off to SearchAgent."


# =============================================================== SearchAgent
async def search_resources(
    ctx: Context, query: str, platforms: Optional[list[str]] = None, max_results_per_platform: int = 5
) -> str:
    """Search educational platforms for one query and store the new (deduplicated) results. platforms can include github, huggingface, web, youtube (youtube only works if available). If platforms is omitted, all available platforms are searched. Returns a short summary; the EvaluationAgent will review the results."""
    async with locked(ctx):
        state = await get_state(ctx)
        if not state["profile"]:
            return "ERROR: no saved profile yet."
        if state["searches_this_cycle"] >= config.max_searches_per_cycle:
            return (f"SEARCH BUDGET USED ({config.max_searches_per_cycle}/{config.max_searches_per_cycle}) "
                    f"for cycle {state['cycle']}. Hand off to EvaluationAgent now.")
        usable = available_platforms()
        wanted = [p.lower().strip() for p in platforms] if platforms else usable
        skipped = [p for p in wanted if p not in usable]
        chosen = [p for p in wanted if p in usable] or usable
        n = max(1, min(int(max_results_per_platform), 10))

        found = await asyncio.to_thread(search_all_platforms, query, chosen, n)
        seen = {normalize_url(r["url"]) for r in state["resources"].values()}
        new = dedupe_resources(found, seen)
        for r in new:
            r["id"] = f"R{state['next_id']}"
            state["next_id"] += 1
            state["resources"][r["id"]] = r
            state["pending"].append(r["id"])
        state["queries"].append(query)
        state["searches_this_cycle"] += 1

        if new:
            await asyncio.to_thread(result_save, new, config.resources_csv)
            try:
                await asyncio.to_thread(_resource_store().add, new, state["run_id"])
            except Exception as exc:  # noqa: BLE001 - RAG indexing must not break searching
                log.warning("Could not index resources: %s", exc)
        await put_state(ctx, state)

        note = f" Skipped unavailable platforms: {skipped}." if skipped else ""
        sample = "\n".join(f"  {_short(r, 80)}" for r in new[:4])
        return (f"Found {len(found)} results, {len(new)} new after removing duplicates.{note} "
                f"Searches this cycle: {state['searches_this_cycle']}/{config.max_searches_per_cycle}.\n{sample}")


# =============================================================== EvaluationAgent
class EvaluationInput(BaseModel):
    id: str = Field(description="Resource id such as R3, exactly as shown by get_pending_resources")
    relevance: float = Field(ge=0, le=1, description="0-1: how directly it addresses the requested topics")
    coverage: float = Field(ge=0, le=1, description="0-1: how completely it covers the topic(s) it targets")
    level_match: float = Field(ge=0, le=1, description="0-1: fit with the user's level")
    quality: float = Field(ge=0, le=1, description="0-1: source reputation, popularity, clarity")
    topics_covered: list[str] = Field(default_factory=list, description="Exact topic names from the profile that this resource covers")
    issues: list[str] = Field(default_factory=list, description="Problems: duplicate, off-topic, wrong level, paid, unclear")
    reason: str = Field(default="", description="One sentence explaining the scores")


async def get_pending_resources(ctx: Context, limit: int = 8) -> str:
    """Get the next resources waiting for evaluation (id, title, platform, price, URL, description). Judge them, then pass your scores to record_evaluations."""
    state = await get_state(ctx)
    budget = config.max_resources_to_evaluate - state["evaluated_this_cycle"]
    if budget <= 0:
        return f"EVALUATION BUDGET USED for cycle {state['cycle']}. Call check_topic_coverage now."
    ids = state["pending"][: max(1, min(int(limit), budget))]
    if not ids:
        return "No resources are pending evaluation."
    left = len(state["pending"]) - len(ids)
    return "\n".join(_short(state["resources"][i], 300) for i in ids) + f"\n({left} more pending)"


async def record_evaluations(ctx: Context, evaluations: list[EvaluationInput]) -> str:
    """Record your scores (0-1) for resources returned by get_pending_resources. The system adds the cost score, computes the weighted total and decides pass/fail. Send up to 8 evaluations per call."""
    async with locked(ctx):
        state = await get_state(ctx)
        topics = _topics(state)
        lines = []
        for item in evaluations:
            try:
                ev = EvaluationInput.model_validate(item)
            except Exception as exc:  # noqa: BLE001
                lines.append(f"REJECTED an evaluation (invalid fields: {exc})")
                continue
            res = state["resources"].get(ev.id)
            if not res:
                lines.append(f"[{ev.id}] unknown resource id, skipped")
                continue
            report = score_evaluation(ev.model_dump(), res, topics)
            state["reports"][ev.id] = report
            state["evaluated_this_cycle"] += 1
            if ev.id in state["pending"]:
                state["pending"].remove(ev.id)
            lines.append(f"[{ev.id}] score {report['score']:.2f} -> {'PASS' if report['passed'] else 'FAIL'}")
        await put_state(ctx, state)
        return "\n".join(lines) + f"\nPending: {len(state['pending'])}"


async def check_topic_coverage(ctx: Context) -> str:
    """Check which profile topics are covered by at least one passing resource and get advice on the next step."""
    state = await get_state(ctx)
    topics = _topics(state)
    passed = [r for r in state["reports"].values() if r["passed"]]
    covered, missing = topic_coverage(passed, topics, config.min_resources_per_topic)
    cycle, limit = state["cycle"], config.max_cycles
    if state["pending"] and state["evaluated_this_cycle"] < config.max_resources_to_evaluate:
        advice = "Resources are still pending: evaluate them first."
    elif not missing:
        advice = "All topics are covered: call finalize_evaluation, then hand off to LearningPathAgent."
    elif cycle < limit:
        advice = "Topics are missing and cycles remain: call request_more_search, then hand off to SearchAgent."
    else:
        advice = "Cycle limit reached: call finalize_evaluation (gaps will be reported), then hand off to LearningPathAgent."
    return json.dumps(
        {"covered": covered, "missing": missing, "passed": len(passed),
         "failed": len(state["reports"]) - len(passed), "cycle": f"{cycle}/{limit}", "advice": advice}
    )


async def request_more_search(ctx: Context, missing_topics: list[str], reasons: list[str], new_queries: list[str]) -> str:
    """Start another search cycle: store your feedback (missing topics, why resources failed) and 2-4 improved queries for SearchAgent. Refused when the cycle limit is reached. After it succeeds, hand off to SearchAgent."""
    async with locked(ctx):
        state = await get_state(ctx)
        if state["cycle"] >= config.max_cycles:
            return (f"CYCLE LIMIT REACHED ({state['cycle']}/{config.max_cycles}). No more searching. "
                    "Call finalize_evaluation and hand off to LearningPathAgent.")
        state["cycle"] += 1
        state["searches_this_cycle"] = 0
        state["evaluated_this_cycle"] = 0
        state["feedback"] = {"missing_topics": missing_topics, "reasons": reasons, "suggested_queries": new_queries}
        await put_state(ctx, state)
        return f"OK: starting cycle {state['cycle']}/{config.max_cycles}. Hand off to SearchAgent."


async def finalize_evaluation(ctx: Context) -> str:
    """Rank the resources that passed evaluation (relevance, level match, cost, quality, practical value), group them by topic and save the ranking. Call once evaluation is complete, then hand off to LearningPathAgent."""
    async with locked(ctx):
        state = await get_state(ctx)
        reports = list(state["reports"].values())
        passed_ids = [r["id"] for r in reports if r["passed"]]
        if not passed_ids and state["cycle"] < config.max_cycles:
            return "NOT FINALIZED: nothing has passed yet and cycles remain. Call request_more_search instead."
        approved = [state["resources"][i] for i in passed_ids]
        ranked = rank_resources(approved, reports, state["profile"])
        grouped = group_resources_by_topic(ranked, _topics(state))
        saved_msg = await asyncio.to_thread(ranking_save, ranked, config.ranking_csv)
        state["ranked"], state["grouped"] = ranked, grouped
        await put_state(ctx, state)
        gaps = [t for t, rs in grouped.items() if not rs]
        return f"{saved_msg}. Ranked {len(ranked)} resources. Topics without a resource: {gaps or 'none'}. Now hand off to LearningPathAgent."


# =============================================================== LearningPathAgent
async def get_ranked_resources(ctx: Context) -> str:
    """List the approved, ranked resources grouped by topic (id, rank, score, title, platform, price, URL, duration)."""
    state = await get_state(ctx)
    if not state["ranked"]:
        return "No ranked resources (nothing passed evaluation)."
    lines = []
    for topic, resources in state["grouped"].items():
        lines.append(f"TOPIC: {topic}" + ("" if resources else "  (no resource found)"))
        for r in resources:
            lines.append(f"  #{r['rank']} score {r['final_score']:.2f} {_short(r, 120)} | duration: {r.get('duration') or 'unknown'}")
    return "\n".join(lines)


def _canon(topics: list[str], name: str) -> Optional[str]:
    return {t.lower(): t for t in topics}.get(str(name).strip().lower())


async def save_roadmap(
    ctx: Context,
    prerequisites: dict[str, list[str]],
    resources_by_topic: dict[str, list[str]],
    practice_tasks_by_topic: dict[str, list[str]],
) -> str:
    """Build and save the final roadmap. prerequisites maps each topic to the topics that must be learned first; resources_by_topic maps each topic to up to 3 resource ids taken from get_ranked_resources; practice_tasks_by_topic maps each topic to 2-3 practice tasks. Use the exact topic names from the profile."""
    async with locked(ctx):
        state = await get_state(ctx)
        topics = _topics(state)
        ranked_ids = {r["id"]: r for r in state["ranked"]}
        warnings = []

        order = build_topic_dependencies(topics, sanitize_prerequisites(topics, prerequisites))
        chosen: dict[str, list[dict]] = {}
        for name, ids in (resources_by_topic or {}).items():
            topic = _canon(topics, name)
            if not topic:
                warnings.append(f"unknown topic '{name}' in resources_by_topic")
                continue
            valid = [ranked_ids[i] for i in ids if i in ranked_ids]
            bad = [i for i in ids if i not in ranked_ids]
            if bad:
                warnings.append(f"ignored ids not in the ranked list for '{topic}': {bad}")
            chosen[topic] = valid[:3]
        tasks = {}
        for name, items in (practice_tasks_by_topic or {}).items():
            topic = _canon(topics, name)
            if topic:
                tasks[topic] = [str(t) for t in items][:3]

        stages = []
        for topic in order:
            resources = chosen.get(topic) or state["grouped"].get(topic, [])[:2]
            if topic not in chosen and resources:
                warnings.append(f"no resources chosen for '{topic}'; used the top-ranked ones")
            if not tasks.get(topic):
                warnings.append(f"no practice tasks for '{topic}'")
            stages.append({"topic": topic, "resources": resources, "practice_tasks": tasks.get(topic, [])})

        state["roadmap"] = generate_roadmap(state["profile"], stages)
        await put_state(ctx, state)
        extra = f" Warnings: {'; '.join(warnings)}" if warnings else ""
        return f"SUCCESS: roadmap saved with {len(stages)} stages.{extra}"
