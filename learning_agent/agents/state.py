"""Shared run state, kept in the LlamaIndex workflow `Context` under one key.

Every agent tool reads/writes this dict, so agents hand work to each other through state (not through
long chat messages). It must stay JSON-serialisable.
"""
import asyncio
import uuid

STATE_KEY = "lra_state"
_locks: dict[int, asyncio.Lock] = {}


def new_state(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "run_id": uuid.uuid4().hex,
        "profile": None,
        "cycle": 1,                  # current search/evaluate cycle (1..max_cycles)
        "searches_this_cycle": 0,
        "evaluated_this_cycle": 0,
        "queries": [],               # every query used so far
        "feedback": None,            # evaluator -> searcher feedback for the next cycle
        "next_id": 1,
        "resources": {},             # id -> resource dict
        "pending": [],               # ids waiting for evaluation
        "reports": {},               # id -> evaluation report
        "ranked": [],
        "grouped": {},
        "roadmap": None,
    }


async def _get(ctx, default=None):
    if hasattr(ctx, "store"):
        return await ctx.store.get(STATE_KEY, default=default)
    return await ctx.get(STATE_KEY, default=default)  # older llama-index versions


async def _set(ctx, value) -> None:
    if hasattr(ctx, "store"):
        await ctx.store.set(STATE_KEY, value)
    else:
        await ctx.set(STATE_KEY, value)


async def ensure_state(ctx, user_id: str) -> dict:
    state = await _get(ctx)
    if state is None:
        state = new_state(user_id)
        await _set(ctx, state)
    return state


async def get_state(ctx) -> dict:
    state = await _get(ctx)
    if state is None:
        raise RuntimeError("Run state was not initialised (call ensure_state first).")
    return state


async def put_state(ctx, state: dict) -> None:
    await _set(ctx, state)


def locked(ctx) -> asyncio.Lock:
    """One lock per context: the model may emit parallel tool calls that would race on the state."""
    return _locks.setdefault(id(ctx), asyncio.Lock())
