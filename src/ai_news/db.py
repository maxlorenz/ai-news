from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import duckdb
from loguru import logger

from .models import ClassifiedArticle
from .settings import SETTINGS


def _connect() -> duckdb.DuckDBPyConnection:
    logger.debug(
        "Connecting to MotherDuck database '{}' schema '{}'",
        SETTINGS.motherduck_database,
        SETTINGS.motherduck_schema,
    )

    # Connect using the inline token as requested
    con = duckdb.connect(
        f"md:{SETTINGS.motherduck_database}?motherduck_token={SETTINGS.motherduck_token}"
    )

    # Ensure we're on my_db and schema exists
    con.execute(f"USE {SETTINGS.motherduck_database}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SETTINGS.motherduck_schema}")

    # Create articles table
    full_table = f"{SETTINGS.motherduck_schema}.articles"
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {full_table} (
            url TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            date TIMESTAMP NOT NULL,
            summary TEXT,
            is_interesting BOOLEAN,
            dedup_key TEXT
        );
        """
    )

    # Create openrouter_models table
    models_table = f"{SETTINGS.motherduck_schema}.openrouter_models"
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {models_table} (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            pricing_prompt TEXT NOT NULL,
            context_length INTEGER NOT NULL,
            created INTEGER,
            last_updated TIMESTAMP NOT NULL
        );
        """
    )

    return con


def upsert_articles(articles: Iterable[ClassifiedArticle]) -> int:
    arts: list[ClassifiedArticle] = list(articles)
    if not arts:
        logger.info("No articles to upsert into MotherDuck")
        return 0

    con = _connect()
    full_table = f"{SETTINGS.motherduck_schema}.articles"
    try:
        logger.info("Upserting {} articles into MotherDuck", len(arts))
        con.executemany(
            f"""
            INSERT INTO {full_table} (url, title, source, date, summary, is_interesting, dedup_key)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (url) DO NOTHING
            """,
            [
                (
                    str(a.url),
                    a.title,
                    a.source.value,
                    a.date,
                    a.summary,
                    a.is_interesting,
                    a.dedup_key,
                )
                for a in arts
            ],
        )
        return len(arts)
    finally:
        con.close()


def get_recent_articles(hours: int = 48) -> list[ClassifiedArticle]:
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    con = _connect()
    full_table = f"{SETTINGS.motherduck_schema}.articles"
    try:
        logger.debug("Fetching articles newer than {}", cutoff)
        rows = con.execute(
            f"""
            SELECT url, title, source, date, summary, is_interesting, dedup_key
            FROM {full_table}
            WHERE date >= ?
            """,
            [cutoff],
        ).fetchall()
    finally:
        con.close()

    return [
        ClassifiedArticle(
            url=row[0],
            title=row[1],
            source=row[2],
            date=row[3],
            summary=row[4],
            is_interesting=row[5],
            dedup_key=row[6],
        )
        for row in rows
    ]


def get_all_articles() -> list[ClassifiedArticle]:
    con = _connect()
    full_table = f"{SETTINGS.motherduck_schema}.articles"
    try:
        rows = con.execute(
            f"""
            SELECT url, title, source, date, summary, is_interesting, dedup_key
            FROM {full_table}
            ORDER BY date DESC
            """
        ).fetchall()
    finally:
        con.close()

    return [
        ClassifiedArticle(
            url=row[0],
            title=row[1],
            source=row[2],
            date=row[3],
            summary=row[4],
            is_interesting=row[5],
            dedup_key=row[6],
        )
        for row in rows
    ]


def get_todays_articles() -> list[ClassifiedArticle]:
    """Get articles from today (UTC date) only.

    Returns:
        List of articles where date is today (UTC)
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    con = _connect()
    full_table = f"{SETTINGS.motherduck_schema}.articles"
    try:
        logger.debug("Fetching articles from today (>= {})", today_start)
        rows = con.execute(
            f"""
            SELECT url, title, source, date, summary, is_interesting, dedup_key
            FROM {full_table}
            WHERE date >= ?
            ORDER BY date DESC
            """,
            [today_start],
        ).fetchall()
    finally:
        con.close()

    return [
        ClassifiedArticle(
            url=row[0],
            title=row[1],
            source=row[2],
            date=row[3],
            summary=row[4],
            is_interesting=row[5],
            dedup_key=row[6],
        )
        for row in rows
    ]


def replace_openrouter_models(
    models: list[tuple[str, str, str, int, int | None, datetime]],
) -> int:
    """Replace all OpenRouter models in the database (delete old, insert new).

    Args:
        models: List of tuples (id, name, pricing_prompt, context_length, created, last_updated)

    Returns:
        Number of models inserted
    """
    if not models:
        logger.info("No OpenRouter models to replace")
        return 0

    con = _connect()
    models_table = f"{SETTINGS.motherduck_schema}.openrouter_models"
    try:
        # Delete all existing models
        logger.info("Deleting all existing OpenRouter models")
        con.execute(f"DELETE FROM {models_table}")

        # Insert new models
        logger.info("Inserting {} new OpenRouter models", len(models))
        con.executemany(
            f"""
            INSERT INTO {models_table} (id, name, pricing_prompt, context_length, created, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            models,
        )
        return len(models)
    finally:
        con.close()


def get_available_openrouter_models() -> list[str]:
    """Get list of available OpenRouter model IDs from database.

    Returns:
        List of top 10 model IDs, ordered by created timestamp descending (newest first)
    """
    con = _connect()
    models_table = f"{SETTINGS.motherduck_schema}.openrouter_models"
    try:
        logger.debug("Fetching available OpenRouter models from database")
        rows = con.execute(
            f"""
            SELECT id
            FROM {models_table}
            ORDER BY created DESC NULLS LAST, context_length DESC
            LIMIT 10
            """
        ).fetchall()
        model_ids = [row[0] for row in rows]
        logger.info(
            "Found {} OpenRouter models in database (limited to 10)", len(model_ids)
        )
        return model_ids
    finally:
        con.close()
