"""Gradio UI:  python -m learning_agent.app"""
from __future__ import annotations

import gradio as gr

from learning_agent.assistant import LearningAssistant
from learning_agent.llm import configure_llama_index

_sessions: dict[str, LearningAssistant] = {}


def build_app() -> gr.Blocks:
    configure_llama_index()

    async def respond(message: str, history: list, user_id: str):
        user_id = (user_id or "demo_user").strip()
        if user_id not in _sessions:
            _sessions[user_id] = LearningAssistant(user_id)
        assistant = _sessions[user_id]

        history = history + [{"role": "user", "content": message}]
        status = {"role": "assistant", "content": "⏳ Starting…"}
        history.append(status)
        yield history, ""
        try:
            async for kind, payload in assistant.stream(message):
                if kind == "status":
                    status["content"] = payload
                else:
                    history.remove(status)
                    history.append({"role": "assistant", "content": payload.reply})
                    if payload.done:
                        _sessions.pop(user_id, None)  # next request starts a fresh conversation
                yield history, ""
        except Exception as exc:  # noqa: BLE001
            if status in history:
                history.remove(status)
            history.append({"role": "assistant", "content": f"⚠️ Something went wrong: `{exc}`"})
            _sessions.pop(user_id, None)
            yield history, ""

    with gr.Blocks(title="Learning Resource Finder") as demo:
        gr.Markdown("# 🎓 Personalized Learning Resource Finder\nTell me what you want to learn, your level, and any preferences (budget, language, platforms).")
        user_id = gr.Textbox(label="User ID (profiles are remembered per ID)", value="demo_user")
        chat = gr.Chatbot(type="messages", height=520)
        box = gr.Textbox(placeholder="e.g. I want to learn RAG with LlamaIndex. I know Python basics. Free resources only.", label="Message")
        box.submit(respond, [box, chat, user_id], [chat, box])
    return demo


if __name__ == "__main__":
    build_app().launch()
