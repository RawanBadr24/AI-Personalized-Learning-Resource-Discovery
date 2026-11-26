# 🎓 Personalized Learning Resource Finder

A **multi-agent** system built with **LlamaIndex** (`AgentWorkflow` + `FunctionAgent`). You describe what you want to learn; agents that reason, call tools and hand off to each other find **free, level-appropriate resources**, rank them, and build a step-by-step roadmap.

- **LLM:** Qwen3 via **Groq** (`qwen/qwen3-32b` by default; change `GROQ_MODEL` in `.env`)
- **Search (free):** GitHub, Hugging Face Hub, DuckDuckGo web, and **YouTube (optional, skipped without a key)**
- **Context:** LlamaIndex `Context` (shared run state + agent memory) **and** RAG with ChromaDB
- **Storage:** ChromaDB for user profiles + embedded resources; CSV for collected and ranked resources
- **UI:** Jupyter/Colab notebook + Gradio

## Architecture

```
            ┌──────────────┐  data_save   ┌──────────────┐        ┌─────────────────┐
 user ◄────►│  InfoAgent   │─────────────►│ SearchAgent  │◄──────►│ EvaluationAgent │
 (clarifying│ profile,level│   handoff    │ queries +    │ handoff│ scores, coverage│
  questions)│ preferences  │              │ search tools │ (loop) │ search-again?   │
            └──────────────┘              └──────────────┘        └────────┬────────┘
                                                                   finalize│(ranking = code)
                                                                           ▼
                                                                  ┌─────────────────┐
                                                                  │LearningPathAgent│
                                                                  │ prerequisites,  │
                                                                  │ tasks, roadmap  │
                                                                  └─────────────────┘
```

| Agent | Decides | Tools |
|---|---|---|
| **InfoAgent** | what to ask, when the profile is complete | `profile_retrieve`, `analyze_learning_goal`, `assess_user_level_and_preferences`, `goal_decomposer`, `data_save` |
| **SearchAgent** | which queries and platforms to use, when to stop | `get_learning_brief`, `search_resources` |
| **EvaluationAgent** | scores each resource, whether to search again or finalize | `get_pending_resources`, `record_evaluations`, `check_topic_coverage`, `find_in_index` (RAG), `request_more_search`, `finalize_evaluation` |
| **LearningPathAgent** | topic order, resource choice, practice tasks | `get_ranked_resources`, `find_in_index` (RAG), `save_roadmap` |

**What is plain code (not an agent):** ranking (weighted math inside `finalize_evaluation`), the scoring/pass-fail math, de-duplication, CSV saving, and the **guards** — the 3-cycle limit, per-cycle search/evaluation budgets, and validation that the roadmap only uses resource ids that exist. Guards live in the tools, so a model cannot talk its way past them. The final markdown (with URLs) is rendered from saved state, not retyped by the LLM, so links cannot be hallucinated.

If an agent stops with plain text mid-pipeline, `LearningAssistant` nudges it to continue (up to `max_nudges`).

### Context

| Type | Where | Used for |
|---|---|---|
| `Context` (workflow) | `agents/state.py` | one shared state dict: profile, resources, reports, cycle counter, ranking, roadmap; also the agents' chat memory (so clarifying questions work across turns) |
| RAG – profiles | `rag/store.py: ProfileStore` | `profile_retrieve` / `data_save` (exact + semantic lookup) |
| RAG – resources | `rag/store.py: ResourceStore` | `find_in_index` tool: topic coverage checks, near-duplicate detection, finding close matches |

## Project layout

```
learning_agent/
  config.py  llm.py  prompts.py  utils.py  render.py  assistant.py  app.py
  agents/   state.py  tools.py  team.py
  rag/      store.py
  tools/    profile.py  search.py  evaluation.py  ranking.py  path.py   # deterministic building blocks
tests/              # pytest, no network / API key needed
notebooks/demo.ipynb
```

## Run it

### Colab
```python
!git clone https://github.com/RawanBadr24/Learning-Resources-Agent.git
%cd learning-resource-agent
!pip install -q -r requirements.txt
```
Add a Colab secret named `GROQ_API_KEY` (🔑 sidebar), then open `notebooks/demo.ipynb`. To keep data across sessions, mount Drive and set `DATA_DIR` to a Drive folder before importing `learning_agent`.

### Local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add GROQ_API_KEY
python -m learning_agent.app                           # Gradio UI
```

## Tests

```bash
pytest -v
```
No keys or internet needed (LLM, search APIs and embeddings are mocked or avoided). `test_stores.py` is skipped if `chromadb` / `llama-index` are missing.

## Configuration (`learning_agent/config.py`)

Model, `max_cycles` (3), `max_searches_per_cycle` (4), `max_resources_to_evaluate` (15 per cycle), `max_agent_iterations`, pass threshold, and scoring weights:

| Evaluation | weight | Ranking | weight |
|---|---|---|---|
| relevance | 0.35 | relevance | 0.30 |
| level match | 0.25 | level match | 0.25 |
| coverage | 0.20 | cost (×1.5 if budget = free) | 0.20 |
| quality | 0.10 | quality | 0.15 |
| cost (rule-based) | 0.10 | practical value (= coverage) | 0.10 |


