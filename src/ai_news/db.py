from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List

import duckdb
from loguru import logger

from .models import ClassifiedArticle
from .settings import SETTINGS


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    date TIMESTAMP NOT NULL,
    summary TEXT,
    is_interesting BOOLEAN,
    dedup_key TEXT
);
"""


def get_connection() -> duckdb.DuckDBPyConnection:
    logger.debug("Opening DuckDB at {}", SETTINGS.duckdb_path)
    con = duckdb.connect(SETTINGS.duckdb_path)
    con.execute(SCHEMA_SQL)
    return con


def upsert_articles(articles: Iterable[ClassifiedArticle]) -> int:
    arts: List[ClassifiedArticle] = list(articles)
    if not arts:
        logger.info("No articles to upsert into DuckDB")
        return 0

    con = get_connection()
    try:
        logger.info("Upserting {} articles into DuckDB", len(arts))
        con.executemany(
            """
            INSERT INTO articles (url, title, source, date, summary, is_interesting, dedup_key)
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


def get_recent_articles(hours: int = 48) -> List[ClassifiedArticle]:
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    con = get_connection()
    try:
        logger.debug("Fetching articles newer than {}", cutoff)
        rows = con.execute(
            "SELECT url, title, source, date, summary, is_interesting, dedup_key FROM articles WHERE date >= ?",
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


def get_all_articles() -> List[ClassifiedArticle]:
    con = get_connection()
    try:
        rows = con.execute(
            "SELECT url, title, source, date, summary, is_interesting, dedup_key FROM articles ORDER BY date DESC",
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
