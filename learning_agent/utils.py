"""LLM helpers: think-tag stripping, JSON extraction, and a single `llm_complete` entry point.

Tools call `llm_json(...)`, which calls `llm_complete(...)`. Tests monkeypatch
`learning_agent.utils.llm_complete`, so no real LLM is needed to test the logic.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from learning_agent.prompts import SYSTEM_BASE

log = logging.getLogger(__name__)
_MISSING = object()


def strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.replace("<think>", "").strip()


def extract_json(text: str) -> Any:
    """Return the first JSON object/array found in `text` (handles code fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", strip_think(text))
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON found in model output")


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def llm_complete(prompt: str, system: str | None = None) -> str:
    """Single chat call to the configured LLM. Returns text with <think> blocks removed."""
    from llama_index.core.llms import ChatMessage

    from learning_agent.llm import get_llm

    messages = []
    if system:
        messages.append(ChatMessage(role="system", content=system))
    messages.append(ChatMessage(role="user", content=prompt))
    response = get_llm().chat(messages)
    return strip_think(response.message.content or "")


def llm_json(prompt: str, system: str | None = SYSTEM_BASE, default: Any = _MISSING) -> Any:
    """Ask the LLM for JSON. Retries once; returns `default` (if given) instead of raising."""
    suffix = "\n\nRespond with ONLY valid JSON. No explanations, no markdown. /no_think"
    last_error: Exception | None = None
    for attempt in range(2):
        extra = "" if attempt == 0 else "\n\nYour previous answer was not valid JSON. Return ONLY JSON."
        try:
            return extract_json(llm_complete(prompt + suffix + extra, system=system))
        except Exception as exc:  # noqa: BLE001 - network/parse errors both handled the same way
            last_error = exc
            log.warning("llm_json attempt %d failed: %s", attempt + 1, exc)
    if default is not _MISSING:
        return default
    raise ValueError(f"LLM did not return valid JSON: {last_error}")
