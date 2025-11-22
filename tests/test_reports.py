from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import HttpUrl

from ai_news.models import ClassifiedArticle, Source
from ai_news.reports import DailyReport, generate_daily_report, select_top_3_articles


@pytest.fixture
def sample_articles():
    """Create sample classified articles for testing."""
    return [
        ClassifiedArticle(
            title="GPT-5 Released with Revolutionary Features",
            url=HttpUrl("https://example.com/gpt5"),
            source=Source.HACKER_NEWS,
            date=datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC),
            summary="OpenAI releases GPT-5 with major improvements",
            is_interesting=True,
            dedup_key="gpt5-release",
        ),
        ClassifiedArticle(
            title="New Research on Transformer Architecture",
            url=HttpUrl("https://example.com/transformers"),
            source=Source.HUGGINGFACE_PAPERS,
            date=datetime(2025, 1, 15, 11, 0, 0, tzinfo=UTC),
            summary="Novel approach to attention mechanisms",
            is_interesting=True,
            dedup_key="transformer-research",
        ),
        ClassifiedArticle(
            title="Claude 4 Announcement",
            url=HttpUrl("https://example.com/claude4"),
            source=Source.META_AI,
            date=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            summary="Anthropic announces Claude 4 with safety improvements",
            is_interesting=True,
            dedup_key="claude4-announcement",
        ),
    ]


def test_select_top_3_with_fewer_articles():
    """Test that select_top_3_articles returns all articles if there are 3 or fewer."""
    articles = [
        ClassifiedArticle(
            title="Article 1",
            url=HttpUrl("https://example.com/1"),
            source=Source.HACKER_NEWS,
            date=datetime.now(UTC),
            is_interesting=True,
        ),
        ClassifiedArticle(
            title="Article 2",
            url=HttpUrl("https://example.com/2"),
            source=Source.HACKER_NEWS,
            date=datetime.now(UTC),
            is_interesting=True,
        ),
    ]

    top_3, reasoning = select_top_3_articles(articles)

    assert len(top_3) == 2
    assert reasoning is None  # No AI selection needed
    assert top_3 == articles


def test_generate_daily_report_with_no_articles():
    """Test that generate_daily_report handles empty article list."""
    report = generate_daily_report([])

    assert isinstance(report, DailyReport)
    assert report.article_count == 0
    assert "No interesting AI news today" in report.report_content
    assert report.top_3_article_urls == []
    assert report.selection_reasoning is None


def test_generate_daily_report_structure(sample_articles):
    """Test that generate_daily_report creates proper report structure."""
    # Mock the external dependencies to avoid real API calls
    with patch("ai_news.reports.select_top_3_articles") as mock_select:
        with patch("ai_news.reports.JinaScraper") as mock_scraper_class:
            with patch("ai_news.reports.summarize_article_content") as mock_summarize:
                # Setup mocks
                mock_select.return_value = (sample_articles, "Test reasoning")
                mock_scraper = MagicMock()
                mock_scraper.scrape_url.return_value = "Scraped content for testing"
                mock_scraper_class.return_value = mock_scraper
                mock_summarize.return_value = "This is a test summary."

                report = generate_daily_report(sample_articles)

                assert isinstance(report, DailyReport)
                assert report.article_count == 3
                assert len(report.report_content) > 0
                assert "Daily AI News Summary" in report.report_content
                assert "All Articles:" in report.report_content
                assert len(report.top_3_article_urls) <= 3


def test_daily_report_model_validation():
    """Test DailyReport model validation."""
    report = DailyReport(
        report_date=datetime.now(UTC),
        article_count=5,
        report_content="Test content",
        top_3_article_urls=[
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ],
        selection_reasoning="Test reasoning",
    )

    assert report.article_count == 5
    assert len(report.top_3_article_urls) == 3
    assert report.selection_reasoning == "Test reasoning"


def test_report_content_includes_article_titles(sample_articles):
    """Test that report content includes article titles."""
    # Mock external dependencies
    with patch("ai_news.reports.select_top_3_articles") as mock_select:
        with patch("ai_news.reports.JinaScraper") as mock_scraper_class:
            with patch("ai_news.reports.summarize_article_content") as mock_summarize:
                mock_select.return_value = (sample_articles, "Test reasoning")
                mock_scraper = MagicMock()
                mock_scraper.scrape_url.return_value = "Scraped content"
                mock_scraper_class.return_value = mock_scraper
                mock_summarize.return_value = "Test summary"

                report = generate_daily_report(sample_articles)

                # Check that all article titles appear in the report
                for article in sample_articles:
                    assert article.title in report.report_content


def test_report_content_markdown_formatting(sample_articles):
    """Test that report uses proper markdown formatting."""
    # Mock external dependencies
    with patch("ai_news.reports.select_top_3_articles") as mock_select:
        with patch("ai_news.reports.JinaScraper") as mock_scraper_class:
            with patch("ai_news.reports.summarize_article_content") as mock_summarize:
                mock_select.return_value = (sample_articles, "Test reasoning")
                mock_scraper = MagicMock()
                mock_scraper.scrape_url.return_value = "Scraped content"
                mock_scraper_class.return_value = mock_scraper
                mock_summarize.return_value = "Test summary"

                report = generate_daily_report(sample_articles)

                # Check for markdown formatting elements
                assert "*" in report.report_content  # Bold text
                assert "[" in report.report_content  # Links
                assert "(" in report.report_content  # Link URLs
                assert "🤖" in report.report_content  # Emojis


def test_top_3_urls_match_selected_articles(sample_articles):
    """Test that top_3_article_urls contains URLs from selected articles."""
    # Mock external dependencies
    with patch("ai_news.reports.select_top_3_articles") as mock_select:
        with patch("ai_news.reports.JinaScraper") as mock_scraper_class:
            with patch("ai_news.reports.summarize_article_content") as mock_summarize:
                mock_select.return_value = (sample_articles, "Test reasoning")
                mock_scraper = MagicMock()
                mock_scraper.scrape_url.return_value = "Scraped content"
                mock_scraper_class.return_value = mock_scraper
                mock_summarize.return_value = "Test summary"

                report = generate_daily_report(sample_articles)

                # All top 3 URLs should be from our sample articles
                sample_urls = {str(art.url) for art in sample_articles}
                for url in report.top_3_article_urls:
                    assert url in sample_urls
