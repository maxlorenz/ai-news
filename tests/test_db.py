"""Tests for database operations."""

import pytest
from datetime import datetime
import duckdb

from ai_news.models import ClassifiedArticle, Source
from ai_news.db import (
    upsert_articles,
    get_recent_articles,
    get_todays_articles,
    replace_openrouter_models,
)


@pytest.fixture
def local_db(tmp_path, monkeypatch):
    """Setup local DuckDB for testing."""
    db_path = tmp_path / "test.db"

    # Mock settings to use local DB
    from ai_news import settings

    monkeypatch.setattr(settings.SETTINGS, "motherduck_token", "")
    monkeypatch.setattr(settings.SETTINGS, "motherduck_database", str(db_path))
    monkeypatch.setattr(settings.SETTINGS, "motherduck_schema", "main")

    # Create test table
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE main.articles (
            url TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            date TIMESTAMP NOT NULL,
            summary TEXT,
            is_interesting BOOLEAN,
            dedup_key TEXT
        )
    """)
    con.close()

    return db_path


@pytest.mark.unit
def test_upsert_articles(local_db):
    """Test upserting articles into database."""
    articles = [
        ClassifiedArticle(
            title="Test Article",
            url="https://test.com/article1",
            source=Source.HACKER_NEWS,
            date=datetime(2025, 11, 22, 8, 0, 0),
            is_interesting=True,
            summary="Test summary",
            dedup_key="test1",
        )
    ]

    count = upsert_articles(articles)
    assert count == 1

    # Try to insert same article again - should do nothing
    count2 = upsert_articles(articles)
    assert count2 == 1


@pytest.mark.unit
def test_get_recent_articles(local_db):
    """Test fetching recent articles."""
    # Insert test articles
    articles = [
        ClassifiedArticle(
            title="Recent Article",
            url="https://test.com/recent",
            source=Source.HACKER_NEWS,
            date=datetime.utcnow(),
            is_interesting=True,
            summary="Recent",
            dedup_key="recent",
        )
    ]
    upsert_articles(articles)

    recent = get_recent_articles(hours=48)
    assert len(recent) == 1
    assert recent[0].title == "Recent Article"


@pytest.mark.unit
def test_get_todays_articles(local_db):
    """Test fetching today's articles."""
    articles = [
        ClassifiedArticle(
            title="Today Article",
            url="https://test.com/today",
            source=Source.HACKER_NEWS,
            date=datetime.utcnow(),
            is_interesting=True,
            summary="Today",
            dedup_key="today",
        )
    ]
    upsert_articles(articles)

    todays = get_todays_articles()
    assert len(todays) == 1
    assert todays[0].title == "Today Article"
