from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic import BaseModel

# Point to project root, not src/
BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
ENV_PATH: Final[Path] = BASE_DIR / ".env"

# Load .env as early as possible
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class AppSettings(BaseModel):
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str | None = None
    openrouter_title: str | None = None

    motherduck_token: str
    motherduck_database: str = "my_db"
    motherduck_schema: str = "ai_articles"

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


# Load from env with no hard-coded defaults
OPENROUTER_API_KEY: Final[str] = os.getenv("OPENROUTER_API_KEY") or ""
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY must be set in .env")

OPENROUTER_HTTP_REFERER: Final[str | None] = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_X_TITLE: Final[str | None] = os.getenv("OPENROUTER_X_TITLE")

MOTHERDUCK_TOKEN: Final[str] = os.getenv("MOTHERDUCK_TOKEN") or ""
if not MOTHERDUCK_TOKEN:
    raise RuntimeError("MOTHERDUCK_TOKEN must be set in .env")

TELEGRAM_BOT_TOKEN: Final[str | None] = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: Final[str | None] = os.getenv("TELEGRAM_CHAT_ID")

SETTINGS: Final[AppSettings] = AppSettings(
    openrouter_api_key=OPENROUTER_API_KEY,
    openrouter_referer=OPENROUTER_HTTP_REFERER,
    openrouter_title=OPENROUTER_X_TITLE,
    motherduck_token=MOTHERDUCK_TOKEN,
    telegram_bot_token=TELEGRAM_BOT_TOKEN,
    telegram_chat_id=TELEGRAM_CHAT_ID,
)


# List of OpenRouter models to randomly choose from (mix of free and paid)
OPENROUTER_MODELS: Final[list[str]] = [
    # Non-free models (paid)
    "google/gemini-2.5-flash",
    "openai/gpt-5-mini",
    "qwen/qwen3-235b-a22b-2507",
    "openai/gpt-oss-120b",
    "openai/gpt-5-nano",
    "mistralai/mistral-small-24b-instruct-2501",
    # Free models
    "x-ai/grok-4.1-fast:free",
    "z-ai/glm-4.5-air:free",
    "moonshotai/kimi-k2:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    "qwen/qwen3-30b-a3b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]
