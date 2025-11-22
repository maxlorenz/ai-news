from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from .db import get_all_articles, get_recent_articles, upsert_articles
from .llm import classify_articles, detect_duplicates
from .sources import fetch_all_sources


async def _async_run() -> None:
    now = datetime.utcnow()
    logger.info("Starting AI news run at {}", now.isoformat())

    # 1. Fetch raw articles from all sources
    raw_articles = await fetch_all_sources(current_date=now)

    # 2. Classify and summarise via LLM
    classified = classify_articles(raw_articles)

    # 3. Load recent articles for duplicate detection (48h)
    recent = get_recent_articles(hours=48)

    # 4. Deduplicate based on URL, title, and LLM dedup key
    unique, duplicates = detect_duplicates(classified, recent)

    for dup in duplicates:
        logger.info("Skipping duplicate: {} ({})", dup.title, dup.url)

    interesting_to_store = [a for a in unique if a.is_interesting]
    logger.info(
        "Articles: {} raw, {} classified interesting, {} unique interesting after dedup",
        len(raw_articles),
        sum(1 for a in classified if a.is_interesting),
        len(interesting_to_store),
    )

    # 5. Upsert into DuckDB
    inserted = upsert_articles(interesting_to_store)
    logger.info("Upserted {} interesting articles into DuckDB", inserted)

    # 6. Log current DB snapshot
    all_articles = get_all_articles()
    logger.info("DuckDB currently holds {} articles", len(all_articles))
    for art in all_articles[:10]:
        logger.info(
            "[DB] {} | {} | {} | {}",
            art.date,
            art.source.value,
            art.title,
            art.url,
        )


def run() -> None:
    asyncio.run(_async_run())
