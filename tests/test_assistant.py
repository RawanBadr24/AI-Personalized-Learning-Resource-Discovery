"""LearningAssistant with a fake workflow: tests the 'nudge' loop and result building (no llama-index needed)."""
from learning_agent import assistant as A
from learning_agent.agents.state import STATE_KEY, new_state
from tests.conftest import FakeCtx


class Ev:  # fake event
    def __init__(self, name, **kw):
        self.__class__ = type(name, (Ev,), {})
        self.__dict__.update(kw)


class FakeHandler:
    def __init__(self, text, events):
        self.text, self.events = text, events

    async def stream_events(self):
        for e in self.events:
            yield e

    def __await__(self):
        async def _r():
            return self.text
        return _r().__await__()


class FakeWorkflow:
    """Each run() call applies the next scripted step to the shared state and returns its text."""

    def __init__(self, ctx, steps):
        self.ctx, self.steps, self.messages = ctx, list(steps), []

    def run(self, user_msg, ctx, max_iterations=None):
        self.messages.append(user_msg)
        mutate, text = self.steps.pop(0)
        mutate(ctx.store.d[STATE_KEY])
        return FakeHandler(text, [Ev("AgentInput", current_agent_name="InfoAgent"), Ev("ToolCall", tool_name="handoff", tool_kwargs={"to_agent": "SearchAgent"})])


def make(steps, max_nudges=2):
    ctx = FakeCtx()
    wf = FakeWorkflow(ctx, steps)
    return A.LearningAssistant("u1", workflow=wf, ctx=ctx, max_nudges=max_nudges), wf


def set_profile(profile):
    def f(st):
        st["profile"] = profile
    return f


def set_roadmap(st):
    st["roadmap"] = {"subject": "s", "level": "beginner", "stages": [
        {"stage": 1, "topic": "Embeddings", "objective": "o", "resources": [], "estimated_hours": 3, "time_basis": "default estimate",
         "practice_tasks": [], "note": "No approved resource found for this topic."}],
        "total_estimated_hours": 3, "estimated_weeks": None}


async def test_clarifying_question_returns_without_nudging(profile):
    assistant, wf = make([(lambda st: None, "What is your current level?")])
    res = await assistant.chat("I want to learn RAG")
    assert res.reply == "What is your current level?" and not res.done and len(wf.messages) == 1


async def test_agent_that_stops_early_is_nudged_until_roadmap(profile):
    def both(st):
        set_profile(profile)(st)
        set_roadmap(st)

    assistant, wf = make([(set_profile(profile), "Searching done."), (lambda st: None, "still thinking"), (both, "Done.")])
    res = await assistant.chat("go")
    assert res.done and res.result["roadmap"] and "Step-by-step roadmap" in res.reply
    assert wf.messages == ["go", A.NUDGE, A.NUDGE]


async def test_gives_up_after_max_nudges_with_a_warning(profile):
    assistant, wf = make([(set_profile(profile), "a"), (lambda st: None, "b"), (lambda st: None, "c")], max_nudges=2)
    res = await assistant.chat("go")
    assert not res.done and "stopped before finishing" in res.reply and len(wf.messages) == 3


async def test_status_events_are_reported(profile):
    assistant, _ = make([(lambda st: (set_profile(profile)(st), set_roadmap(st)), "ok")])
    seen = []
    await assistant.chat("go", on_status=seen.append)
    assert seen == ["🤖 InfoAgent is working…", "➡️ Handing off to SearchAgent"]


def test_describe_event_search_and_other_tools():
    assert "Searching: rag" in A.describe_event(Ev("ToolCall", tool_name="search_resources", tool_kwargs={"query": "rag"}))
    assert A.describe_event(Ev("ToolCall", tool_name="record_evaluations", tool_kwargs={})) == "🔧 record_evaluations"
    assert A.describe_event(Ev("Whatever")) is None
