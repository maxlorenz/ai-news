from __future__ import annotations

from pathlib import Path
from typing import Final, List

from dotenv import load_dotenv
from pydantic import BaseModel
import os


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env"

# Load .env as early as possible
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class AppSettings(BaseModel):
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str | None = None
    openrouter_title: str | None = None
    duckdb_path: str = str(BASE_DIR / "ai_news.duckdb")


SETTINGS: Final[AppSettings] = AppSettings(
    openrouter_api_key=os.getenv(
        "OPENROUTER_API_KEY",
        "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",
    ),
    openrouter_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
    openrouter_title=os.getenv("OPENROUTER_X_TITLE"),
)


# List of free models to randomly choose from
FREE_OPENROUTER_MODELS: Final[List[str]] = [
    "x-ai/grok-4.1-fast:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "moonshotai/kimi-k2:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    "qwen/qwen3-30b-a3b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]
