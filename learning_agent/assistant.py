"""LearningAssistant: runs the agent team for one user, streams progress, and renders the result.

One instance = one conversation. The shared `Context` keeps the chat memory and the run state, so if
InfoAgent asks a clarifying question, the next `chat()` call simply continues where it left off.
"""
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional

from learning_agent.agents.state import ensure_state, get_state
from learning_agent.config import config
from learning_agent.render import format_final_response
from learning_agent.utils import strip_think

NUDGE = ("Continue with your current step and hand off to the next agent when it is done. "
         "Do not stop until the roadmap has been saved with save_roadmap.")


@dataclass
class ChatResult:
    reply: str                 # markdown to show the user
    done: bool                 # True once the roadmap exists
    result: Optional[dict]     # structured result (profile, ranked, roadmap, ...) when done


def build_result(state: dict) -> dict:
    roadmap = state["roadmap"]
    reports = list(state["reports"].values())
    return {
        "profile": state["profile"],
        "ranked": state["ranked"],
        "grouped": state["grouped"],
        "roadmap": roadmap,
        "coverage_gaps": [s["topic"] for s in roadmap["stages"] if not s["resources"]],
        "cycles_used": state["cycle"],
        "evaluated": len(reports),
        "rejected": sum(1 for r in reports if not r["passed"]),
    }


def describe_event(ev) -> Optional[str]:
    """Human-readable progress line for a workflow event (duck-typed, so no llama-index import)."""
    kind = type(ev).__name__
    if kind == "AgentInput":
        return f"🤖 {getattr(ev, 'current_agent_name', 'Agent')} is working…"
    if kind == "ToolCall":
        name = getattr(ev, "tool_name", "")
        kwargs = getattr(ev, "tool_kwargs", {}) or {}
        if name == "handoff":
            return f"➡️ Handing off to {kwargs.get('to_agent', '?')}"
        if name == "search_resources":
            return f"🔎 Searching: {kwargs.get('query', '')}"
        return f"🔧 {name}"
    return None


class LearningAssistant:
    def __init__(self, user_id: str, workflow=None, ctx=None, max_nudges: Optional[int] = None):
        self.user_id = user_id
        self.max_nudges = config.max_nudges if max_nudges is None else max_nudges
        if workflow is None:
            from learning_agent.agents.team import build_team

            workflow = build_team(user_id)
        if ctx is None:
            from llama_index.core.workflow import Context

            ctx = Context(workflow)
        self.workflow, self.ctx = workflow, ctx

    async def state(self) -> dict:
        return await ensure_state(self.ctx, self.user_id)

    async def stream(self, message: str) -> AsyncIterator[tuple]:
        """Yields ("status", text) while agents work, then ("final", ChatResult)."""
        await self.state()
        msg, nudges, reply = message, 0, ""
        while True:
            handler = self.workflow.run(user_msg=msg, ctx=self.ctx, max_iterations=config.max_agent_iterations)
            async for ev in handler.stream_events():
                status = describe_event(ev)
                if status:
                    yield ("status", status)
            reply = strip_think(str(await handler))
            state = await get_state(self.ctx)
            waiting_for_user = state["profile"] is None
            if state["roadmap"] or waiting_for_user or nudges >= self.max_nudges:
                break
            nudges += 1  # an agent stopped with plain text mid-pipeline: push it to continue
            msg = NUDGE

        if state["roadmap"]:
            result = build_result(state)
            yield ("final", ChatResult(format_final_response(result), True, result))
        elif state["profile"] is not None:
            note = "\n\n⚠️ The agents stopped before finishing the roadmap. Try again or rephrase your request."
            yield ("final", ChatResult(reply + note, False, None))
        else:
            yield ("final", ChatResult(reply, False, None))

    async def chat(self, message: str, on_status: Optional[Callable[[str], Awaitable | None]] = None) -> ChatResult:
        """Convenience wrapper around stream(): returns the final ChatResult, optionally reporting progress."""
        final = None
        async for kind, payload in self.stream(message):
            if kind == "status" and on_status:
                maybe = on_status(payload)
                if hasattr(maybe, "__await__"):
                    await maybe
            elif kind == "final":
                final = payload
        return final
