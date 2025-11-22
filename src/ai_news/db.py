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
            ON CONFLICT (url) DO UPDATE SET
                title = excluded.title,
                source = excluded.source,
                date = excluded.date,
                summary = excluded.summary,
                is_interesting = excluded.is_interesting,
                dedup_key = excluded.dedup_key
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
