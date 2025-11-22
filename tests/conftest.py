"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def hackernews_html(fixtures_dir: Path) -> str:
    """Load HackerNews HTML fixture."""
    return (fixtures_dir / "hackernews_sample.html").read_text()


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    """Create a temporary DuckDB file for testing."""
    db_file = tmp_path / "test.db"
    return str(db_file)
