"""Tests for source scraping functions."""

import pytest
from datetime import datetime
from bs4 import BeautifulSoup

from ai_news.sources import _parse_hacker_news_html
from ai_news.models import Source


@pytest.mark.unit
def test_parse_hacker_news_html(hackernews_html: str):
    """Test HackerNews HTML parsing."""
    current_date = datetime(2025, 11, 22, 8, 0, 0)
    articles = _parse_hacker_news_html(hackernews_html, current_date)

    assert len(articles) == 2
    assert articles[0].title == "Superman copy found in mum's attic"
    assert articles[0].source == Source.HACKER_NEWS
    assert str(articles[0].url) == "https://www.bbc.com/news/articles/test1"

    assert articles[1].title == "OpenAI announces GPT-5"
    assert str(articles[1].url) == "https://openai.com/gpt5"


@pytest.mark.unit
def test_parse_empty_html():
    """Test parsing empty HTML returns empty list."""
    from ai_news.sources import _parse_hacker_news_html

    empty_html = "<html><body></body></html>"
    current_date = datetime(2025, 11, 22, 8, 0, 0)
    articles = _parse_hacker_news_html(empty_html, current_date)

    assert articles == []
