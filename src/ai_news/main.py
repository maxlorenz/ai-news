from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from loguru import logger

from .db import (
    get_all_articles,
    get_recent_articles,
    get_todays_articles,
    replace_openrouter_models,
    upsert_articles,
)
from .llm import classify_articles, detect_duplicates
from .openrouter_models import (
    fetch_openrouter_models,
    filter_free_text_models,
    get_model_data_for_db,
)
from .sources import fetch_all_sources
from .telegram import send_daily_summary


async def _async_run() -> None:
    now = datetime.now(UTC)
    logger.info("Starting AI news run at {}", now.isoformat())

    # 0. Fetch and replace OpenRouter models
    try:
        logger.info("=== FETCHING OPENROUTER MODELS ===")
        all_models = fetch_openrouter_models()
        free_text_models = filter_free_text_models(all_models)
        model_data = get_model_data_for_db(free_text_models)
        replaced_count = replace_openrouter_models(model_data)
        logger.info(
            "Replaced all models with {} new free OpenRouter models in database",
            replaced_count,
        )
    except Exception as e:
        logger.error("Failed to fetch OpenRouter models: {}", e)
        logger.warning("Continuing with existing models in database")

    # 1. Fetch raw articles from all sources
    raw_articles = await fetch_all_sources(current_date=now)

    # Log what we fetched
    logger.info("=== FETCHED ARTICLES ===")
    for idx, art in enumerate(raw_articles[:10]):  # Show first 10
        logger.info(
            "[{}] {} | {} | {}",
            idx,
            art.source.value,
            art.title,
            art.url,
        )
    if len(raw_articles) > 10:
        logger.info("... and {} more articles", len(raw_articles) - 10)

    # 2. Group articles by source
    from collections import defaultdict

    by_source: dict[str, list] = defaultdict(list)
    for art in raw_articles:
        by_source[art.source.value].append(art)

    logger.info("Grouped into {} sources", len(by_source))

    # 3. Classify each source independently (one LLM call per source)
    classified = []
    for source_name, articles in by_source.items():
        logger.info("Classifying {} articles from {}", len(articles), source_name)
        source_classified = classify_articles(articles)
        classified.extend(source_classified)

        # Log interesting articles from this source
        interesting_count = sum(1 for a in source_classified if a.is_interesting)
        if interesting_count > 0:
            logger.info(
                "Found {} interesting articles from {}", interesting_count, source_name
            )
            for art in source_classified:
                if art.is_interesting:
                    logger.info(
                        "  ✓ {} | {}",
                        art.title,
                        art.summary,
                    )

    # Log classification summary
    logger.info("=== CLASSIFICATION SUMMARY ===")
    total_interesting = sum(1 for a in classified if a.is_interesting)
    logger.info(
        "Total: {} interesting out of {} articles", total_interesting, len(classified)
    )
    if total_interesting == 0:
        logger.warning("No articles classified as interesting!")

    # 4. Load recent articles for duplicate detection (48h)
    recent = get_recent_articles(hours=48)

    # 5. Deduplicate based on URL, title, and LLM dedup key
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

    # 6. Upsert into MotherDuck
    inserted = upsert_articles(interesting_to_store)
    logger.info("Upserted {} interesting articles into MotherDuck", inserted)

    # 7. Log current snapshot
    all_articles = get_all_articles()
    logger.info("MotherDuck currently holds {} articles", len(all_articles))
    for art in all_articles[:10]:
        logger.info(
            "[DB] {} | {} | {} | {}",
            art.date,
            art.source.value,
            art.title,
            art.url,
        )

    # 8. Send daily Telegram summary (only today's articles)
    try:
        logger.info("=== SENDING TELEGRAM SUMMARY ===")
        todays_articles = get_todays_articles()
        if todays_articles:
            logger.info(
                "Sending Telegram summary for {} articles from today",
                len(todays_articles),
            )
            send_daily_summary(todays_articles)
            logger.info("Telegram summary sent successfully")
        else:
            logger.info("No articles from today to send via Telegram")
    except Exception as e:
        logger.error("Failed to send Telegram summary: {}", e)


def run() -> None:
    asyncio.run(_async_run())
