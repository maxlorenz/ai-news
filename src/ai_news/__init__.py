from __future__ import annotations

from loguru import logger

from .main import run


def main() -> None:
    """Entry point for the ai-news CLI script."""
    logger.add("ai_news.log", rotation="1 MB", retention=5)
    run()
