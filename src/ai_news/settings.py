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
