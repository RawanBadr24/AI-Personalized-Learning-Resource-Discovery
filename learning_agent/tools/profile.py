"""Agent 1 tools: information gathering / user profile."""
from __future__ import annotations

from learning_agent.utils import llm_json


def _store():
    from learning_agent.rag.store import get_profile_store

    return get_profile_store()


def profile_retrieve(user_id: str) -> dict:
    """Retrieve the user's previously saved learning profile. Returns {} if none exists."""
    return _store().get(user_id)


def analyze_learning_goal(user_request: str) -> dict:
    """Extract the learning subject, objectives, expected outcomes, and requested topics from the user's request."""
    prompt = f"""Analyze this learning request.
Request: {user_request!r}

Return JSON: {{"subject": str, "objectives": [str], "expected_outcomes": [str], "requested_topics": [str]}}
Use empty strings/lists for anything not stated. Do not guess."""
    data = llm_json(prompt, default={})
    return {
        "subject": data.get("subject", ""),
        "objectives": data.get("objectives", []),
        "expected_outcomes": data.get("expected_outcomes", []),
        "requested_topics": data.get("requested_topics", []),
    }


def assess_user_level_and_preferences(
    subject: str, self_reported_level: str, known_topics: list[str], user_message: str = ""
) -> dict:
    """Assess the user's knowledge level, identify knowledge gaps/prerequisites, and extract preferences (budget, platforms, language, resource types)."""
    prompt = f"""Subject: {subject}
Self-reported level: {self_reported_level}
Known topics: {known_topics}
User message (for preferences): {user_message!r}

Return JSON:
{{"level": "beginner|intermediate|advanced",
  "knowledge_gaps": [str], "missing_prerequisites": [str],
  "preferences": {{"budget": "free|any|<amount>", "platforms": [str], "language": str, "resource_types": [str]}}}}
Default budget to "free" and language to "English" only if the user did not say otherwise."""
    data = llm_json(prompt, default={})
    prefs = data.get("preferences") or {}
    return {
        "level": data.get("level") or self_reported_level or "beginner",
        "knowledge_gaps": data.get("knowledge_gaps", []),
        "missing_prerequisites": data.get("missing_prerequisites", []),
        "preferences": {
            "budget": prefs.get("budget") or "free",
            "platforms": prefs.get("platforms") or [],
            "language": prefs.get("language") or "English",
            "resource_types": prefs.get("resource_types") or [],
        },
    }


def goal_decomposer(subject: str, learning_objective: str, current_level: str) -> list[str]:
    """Break the main learning goal into 3-6 smaller topics/subtopics suited to the user's current level."""
    prompt = f"""Break this learning goal into 3-6 short, ordered topics.
Subject: {subject}
Objective: {learning_objective}
Current level: {current_level}
Return a JSON list of short topic names (2-5 words each)."""
    data = llm_json(prompt, default=[])
    return [str(t) for t in data][:8] if isinstance(data, list) else []


def data_save(user_profile: dict) -> str:
    """Save the user's learning profile. Returns a success or failure message."""
    try:
        saved = _store().save(user_profile)
        return f"SUCCESS: profile saved for user '{saved['user_id']}'."
    except Exception as exc:  # noqa: BLE001
        return f"FAILURE: could not save profile ({exc})."
