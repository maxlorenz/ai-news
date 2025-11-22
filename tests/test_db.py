"""Tests for database operations."""

from datetime import datetime

import duckdb
import pytest

from ai_news.db import (
    get_recent_articles,
    get_todays_articles,
    upsert_articles,
)
from ai_news.models import ClassifiedArticle, Source


class MockConnection:
    """Wrapper that prevents the real connection from being closed."""

    def __init__(self, real_conn):
        self._conn = real_conn

    def __getattr__(self, name):
        if name == "close":
            return lambda: None  # No-op close
        return getattr(self._conn, name)


@pytest.fixture
def local_db(monkeypatch):
    """Setup in-memory DuckDB for testing."""
    # Create an in-memory connection
    test_conn = duckdb.connect(":memory:")

    # Create schema and tables
    test_conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    test_conn.execute("""
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
    test_conn.execute("""
        CREATE TABLE main.openrouter_models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            pricing_prompt TEXT NOT NULL,
            context_length INTEGER NOT NULL,
            created INTEGER,
            last_updated TIMESTAMP NOT NULL
        )
    """)

    # Wrap the connection to prevent closing
    wrapped_conn = MockConnection(test_conn)

    # Mock the _connect function to return our wrapped connection
    import ai_news.db

    def mock_connect():
        return wrapped_conn

    monkeypatch.setattr(ai_news.db, "_connect", mock_connect)

    # Mock settings
    from ai_news import settings

    monkeypatch.setattr(settings.SETTINGS, "motherduck_schema", "main")

    yield test_conn

    # Cleanup
    test_conn.close()


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

    # Verify only 1 article in DB
    result = local_db.execute("SELECT COUNT(*) FROM main.articles").fetchone()
    assert result[0] == 1


@pytest.mark.unit
def test_get_recent_articles(local_db):
    """Test fetching recent articles."""
    # Insert test articles
    articles = [
        ClassifiedArticle(
            title="Recent Article",
            url="https://test.com/recent",
            source=Source.HACKER_NEWS,
            date=datetime.now(),
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
            date=datetime.now(),
            is_interesting=True,
            summary="Today",
            dedup_key="today",
        )
    ]
    upsert_articles(articles)

    todays = get_todays_articles()
    assert len(todays) == 1
    assert todays[0].title == "Today Article"
