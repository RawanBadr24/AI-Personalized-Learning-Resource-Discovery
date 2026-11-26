"""Central configuration. Everything tunable lives here (or in .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PLATFORMS = ("github", "huggingface", "web", "youtube")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass
class Config:
    # --- LLM (Groq, OpenAI-compatible endpoint) ---
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "qwen/qwen3-32b"))
    groq_base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.2

    # --- Embeddings (local, free) ---
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # --- Optional search keys ---
    youtube_api_key: str = field(default_factory=lambda: _env("YOUTUBE_API_KEY"))
    github_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))

    # --- Storage ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))

    # --- Search / evaluation loop ---
    max_cycles: int = 3
    max_results_per_platform: int = 5
    max_resources_to_evaluate: int = 15  # per cycle (keeps Groq rate limits happy)
    max_searches_per_cycle: int = 4      # search_resources calls allowed per cycle
    max_agent_iterations: int = 80       # hard cap on agent steps for one run
    max_nudges: int = 6                  # times we re-prompt an agent that stopped early
    pass_threshold: float = 0.6
    min_resources_per_topic: int = 1
    default_platforms: tuple = PLATFORMS

    # --- Scoring weights (each set is normalised, so they need not sum to 1) ---
    eval_weights: dict = field(
        default_factory=lambda: {
            "relevance": 0.35,
            "level_match": 0.25,
            "coverage": 0.20,
            "quality": 0.10,
            "cost": 0.10,
        }
    )
    ranking_weights: dict = field(
        default_factory=lambda: {
            "relevance": 0.30,
            "level_match": 0.25,
            "cost": 0.20,
            "quality": 0.15,
            "practical": 0.10,  # practical value = coverage score from evaluation
        }
    )
    default_hours_per_topic: float = 3.0

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def resources_csv(self) -> Path:
        return self.data_dir / "resources.csv"

    @property
    def ranking_csv(self) -> Path:
        return self.data_dir / "ranked_resources.csv"


config = Config()
