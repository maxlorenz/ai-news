from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from loguru import logger

from .db import (
    get_all_articles,
    get_recent_article_urls,
    get_recent_articles,
    get_todays_articles,
    replace_openrouter_models,
    save_report,
    upsert_articles,
)
from .llm import classify_articles, detect_duplicates
from .openrouter_models import (
    fetch_openrouter_models,
    filter_free_text_models,
    get_model_data_for_db,
)
from .scraper import JinaScraper
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

    # 0.5. Load recent article URLs for deduplication
    logger.info("=== LOADING RECENT URLS FOR DEDUPLICATION ===")
    existing_urls = get_recent_article_urls(limit=1000)

    # 1. Fetch raw articles from all sources
    raw_articles = await fetch_all_sources(current_date=now)

    # 1.2. Filter out already-processed articles
    logger.info("=== FILTERING DUPLICATE ARTICLES ===")
    initial_count = len(raw_articles)
    raw_articles = [art for art in raw_articles if str(art.url) not in existing_urls]
    filtered_count = initial_count - len(raw_articles)
    logger.info(
        f"Filtered out {filtered_count} already-processed articles ({len(raw_articles)} remaining from {initial_count} total)"
    )

    if len(raw_articles) == 0:
        logger.info("No new articles to process, exiting")
        return

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

    # 1.5. Enrich Apple ML articles with accurate dates (they default to today)
    from .models import ClassifiedArticle, Source

    apple_ml_articles = [a for a in raw_articles if a.source == Source.APPLE_ML]
    if apple_ml_articles:
        logger.info("=== ENRICHING APPLE ML DATES ===")
        scraper = JinaScraper()
        logger.info(
            "Enriching {} Apple ML articles with dates from Jina scraper",
            len(apple_ml_articles),
        )
        # Convert Article to ClassifiedArticle for enrichment
        apple_ml_classified = [
            ClassifiedArticle(
                url=a.url,
                title=a.title,
                source=a.source,
                date=a.date,
                summary=a.summary,
                is_interesting=False,
            )
            for a in apple_ml_articles
        ]
        enriched_apple_ml = await scraper.enrich_articles_with_dates_async(
            apple_ml_classified
        )

        # Update the dates in raw_articles
        apple_ml_urls = {str(a.url): a for a in enriched_apple_ml}
        for idx, art in enumerate(raw_articles):
            if art.source == Source.APPLE_ML and str(art.url) in apple_ml_urls:
                enriched = apple_ml_urls[str(art.url)]
                raw_articles[idx] = art.model_copy(update={"date": enriched.date})

        logger.info("Finished enriching Apple ML article dates")

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

    # 3.5. Enrich interesting articles with accurate dates using Jina scraper
    logger.info("=== ENRICHING ARTICLES WITH DATES ===")
    interesting_articles = [a for a in classified if a.is_interesting]
    if interesting_articles:
        scraper = JinaScraper()
        logger.info(
            "Enriching {} interesting articles with dates from Jina scraper",
            len(interesting_articles),
        )
        enriched_articles = await scraper.enrich_articles_with_dates_async(
            interesting_articles
        )

        # Replace the interesting articles in classified list with enriched versions
        enriched_map = {str(art.url): art for art in enriched_articles}
        new_classified = []
        for art in classified:
            if art.is_interesting and str(art.url) in enriched_map:
                new_classified.append(enriched_map[str(art.url)])
            else:
                new_classified.append(art)
        classified = new_classified

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
            # Generate report and send via Telegram
            report = send_daily_summary(todays_articles)

            # Store report in database
            logger.info("Storing report in MotherDuck")
            report_id = save_report(
                report_date=report.report_date,
                article_count=report.article_count,
                report_content=report.report_content,
                top_3_article_urls=report.top_3_article_urls,
                selection_reasoning=report.selection_reasoning,
            )
            logger.info(f"Report stored in database with ID: {report_id}")
            logger.info("Telegram summary sent successfully")
        else:
            logger.info("No articles from today to send via Telegram")
    except Exception as e:
        logger.error("Failed to send Telegram summary: {}", e)


def run() -> None:
    asyncio.run(_async_run())
