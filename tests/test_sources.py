"""Tests for source scraping functions."""

from datetime import datetime

import pytest

from ai_news.sources import parse_date_string


@pytest.mark.unit
def test_parse_date_string_formats():
    """Test date string parsing with various formats."""
    # Test standard format
    result = parse_date_string("November 22, 2025")
    assert result == datetime(2025, 11, 22)

    # Test short month
    result = parse_date_string("Nov 22, 2025")
    assert result == datetime(2025, 11, 22)

    # Test ISO format
    result = parse_date_string("2025-11-22")
    assert result == datetime(2025, 11, 22)

    # Test invalid format
    result = parse_date_string("invalid date")
    assert result is None

    # Test empty string
    result = parse_date_string("")
    assert result is None


@pytest.mark.unit
def test_parse_date_string_with_extra_text():
    """Test parsing dates with extra text around them."""
    result = parse_date_string("Posted on Nov 22, 2025")
    assert result == datetime(2025, 11, 22)
