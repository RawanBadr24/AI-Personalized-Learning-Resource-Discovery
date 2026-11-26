import pytest

from learning_agent.utils import clamp01, extract_json, llm_json, strip_think


def test_strip_think_removes_reasoning():
    assert strip_think("<think>hmm\nlong</think>\nHello") == "Hello"


def test_strip_think_handles_unopened_close_tag():
    assert strip_think("reasoning...</think>answer") == "answer"


def test_extract_json_from_fenced_text_with_chatter():
    text = 'Sure! ```json\n{"a": [1, 2], "b": "x"}\n``` hope it helps'
    assert extract_json(text) == {"a": [1, 2], "b": "x"}


def test_extract_json_list_after_think():
    assert extract_json('<think>{"no": 1}</think> ["q1", "q2"]') == ["q1", "q2"]


def test_extract_json_raises_when_missing():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_clamp01():
    assert clamp01(1.7) == 1.0 and clamp01(-3) == 0.0 and clamp01("0.4") == 0.4 and clamp01("abc") == 0.0


def test_llm_json_retries_then_succeeds(fake_llm):
    answers = iter(["not json", '{"ok": true}'])
    fake_llm.handler = lambda p: next(answers)
    assert llm_json("x") == {"ok": True}
    assert len(fake_llm.calls) == 2


def test_llm_json_returns_default_after_failures(fake_llm):
    fake_llm.handler = lambda p: "still not json"
    assert llm_json("x", default={"fallback": 1}) == {"fallback": 1}


def test_llm_json_raises_without_default(fake_llm):
    fake_llm.handler = lambda p: "nope"
    with pytest.raises(ValueError):
        llm_json("x")
