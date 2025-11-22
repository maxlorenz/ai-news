from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime

import datefinder
import httpx
from dateutil import parser as dateutil_parser
from loguru import logger
from pydantic import BaseModel, Field

from .llm import _choose_model
from .llm import _make_client as _make_llm_client
from .models import Article, ClassifiedArticle


class JinaScraper:
    """Handles article scraping and date extraction using Jina.ai Reader API."""

    def __init__(self, timeout: float = 30.0, max_content_length: int = 8000):
        self.timeout = timeout
        self.max_content_length = max_content_length

    def scrape_url(self, url: str) -> str:
        """Scrape article content using Jina.ai Reader API (sync).

        Args:
            url: The URL to scrape

        Returns:
            The scraped content as markdown text
        """
        jina_url = f"https://r.jina.ai/{url}"

        try:
            logger.info(f"Scraping article with Jina.ai: {url}")
            response = httpx.get(jina_url, timeout=self.timeout)
            response.raise_for_status()
            content = response.text

            # Limit content to avoid token limits
            if len(content) > self.max_content_length:
                content = (
                    content[: self.max_content_length] + "\n\n[Content truncated...]"
                )

            logger.info(f"Successfully scraped {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"Failed to scrape article {url}: {e}")
            return f"Failed to scrape content: {str(e)}"

    async def scrape_url_async(self, url: str) -> str:
        """Scrape article content using Jina.ai Reader API (async).

        Args:
            url: The URL to scrape

        Returns:
            The scraped content as markdown text
        """
        jina_url = f"https://r.jina.ai/{url}"

        try:
            logger.info(f"Scraping article with Jina.ai: {url}")
            async with httpx.AsyncClient() as client:
                response = await client.get(jina_url, timeout=self.timeout)
                response.raise_for_status()
                content = response.text

                # Limit content to avoid token limits
                if len(content) > self.max_content_length:
                    content = (
                        content[: self.max_content_length]
                        + "\n\n[Content truncated...]"
                    )

                logger.info(f"Successfully scraped {len(content)} characters")
                return content
        except Exception as e:
            logger.error(f"Failed to scrape article {url}: {e}")
            return f"Failed to scrape content: {str(e)}"

    def extract_date_from_content(self, content: str) -> datetime | None:
        """Extract publication date from article content using multiple methods.

        Args:
            content: The article content (markdown text from Jina)

        Returns:
            Extracted datetime or None if no date found
        """
        if not content:
            return None

        # Method 1: Try dateutil parser with fuzzy matching on first few lines
        # This is more reliable than datefinder for article dates
        search_text = content[:3000]  # Increased to cover more content
        lines = search_text.split("\n")[:60]  # Check first 60 lines

        for line in lines:
            line_stripped = line.strip()
            # Skip very short lines unless they contain a month name
            has_month = any(
                month in line
                for month in [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Sept",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
            )

            if len(line_stripped) < 5 and not has_month:
                continue

            try:
                # Use fuzzy parsing to extract dates from text
                parsed_date = dateutil_parser.parse(line, fuzzy=True)
                # Filter out future dates
                now = datetime.now(UTC)
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=UTC)

                if parsed_date <= now:
                    # Additional validation: date should have month name or be a full date
                    # This helps filter out false positives like "1" being parsed as a date

                    # Or has a date pattern like 2024-10-24 or 10/24/2024
                    has_date_pattern = bool(
                        re.search(
                            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}",
                            line,
                        )
                    )

                    if has_month or has_date_pattern:
                        logger.info(
                            f"Extracted date using dateutil: {parsed_date} from line: '{line[:100]}'"
                        )
                        return parsed_date
            except (ValueError, OverflowError):
                continue

        # Method 2: Try datefinder as fallback (looks for dates in text)
        try:
            # Find all dates in the text
            dates = list(datefinder.find_dates(search_text, source=True))

            if dates:
                # Filter out future dates (likely errors)
                now = datetime.now(UTC)
                valid_dates = []
                for item in dates:
                    if isinstance(item, tuple) and len(item) == 2:
                        dt, text = item
                        if isinstance(dt, datetime):
                            # Make timezone-aware if naive for comparison
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            if dt <= now:
                                # Only accept if source text has reasonable length (>10 chars)
                                # This filters out false positives like just "1" or "2"
                                if len(str(text).strip()) > 10:
                                    valid_dates.append((dt, text))

                if valid_dates:
                    # Take the first valid date found (usually publication date)
                    extracted_date, source_text = valid_dates[0]
                    logger.info(
                        f"Extracted date using datefinder: {extracted_date} from text: '{source_text}'"
                    )
                    return extracted_date
        except Exception as e:
            logger.debug(f"datefinder failed: {e}")

        logger.debug("No date found in content using any extraction method")
        return None

        # Method 1: Try dateutil parser with fuzzy matching on first few lines
        # This is more reliable than datefinder for article dates
        search_text = content[:2000]
        lines = search_text.split("\n")[:30]  # Check first 30 lines

        for line in lines:
            # Skip very short lines that are likely navigation/menus
            if len(line.strip()) < 10:
                continue

            try:
                # Use fuzzy parsing to extract dates from text
                parsed_date = dateutil_parser.parse(line, fuzzy=True)
                # Filter out future dates
                now = datetime.now(UTC)
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=UTC)

                if parsed_date <= now:
                    # Additional validation: date should have month name or be a full date
                    # This helps filter out false positives like "1" being parsed as a date
                    has_month_name = any(
                        month in line
                        for month in [
                            "January",
                            "February",
                            "March",
                            "April",
                            "May",
                            "June",
                            "July",
                            "August",
                            "September",
                            "October",
                            "November",
                            "December",
                            "Jan",
                            "Feb",
                            "Mar",
                            "Apr",
                            "May",
                            "Jun",
                            "Jul",
                            "Aug",
                            "Sep",
                            "Sept",
                            "Oct",
                            "Nov",
                            "Dec",
                        ]
                    )

                    # Or has a date pattern like 2024-10-24 or 10/24/2024
                    has_date_pattern = bool(
                        re.search(
                            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}",
                            line,
                        )
                    )

                    if has_month_name or has_date_pattern:
                        logger.info(
                            f"Extracted date using dateutil: {parsed_date} from line: '{line[:100]}'"
                        )
                        return parsed_date
            except (ValueError, OverflowError):
                continue

        # Method 2: Try datefinder as fallback (looks for dates in text)
        try:
            # Find all dates in the text
            dates = list(datefinder.find_dates(search_text, source=True))

            if dates:
                # Filter out future dates (likely errors)
                now = datetime.now(UTC)
                valid_dates = []
                for item in dates:
                    if isinstance(item, tuple) and len(item) == 2:
                        dt, text = item
                        if isinstance(dt, datetime):
                            # Make timezone-aware if naive for comparison
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            if dt <= now:
                                # Only accept if source text has reasonable length (>10 chars)
                                # This filters out false positives like just "1" or "2"
                                if len(str(text).strip()) > 10:
                                    valid_dates.append((dt, text))

                if valid_dates:
                    # Take the first valid date found (usually publication date)
                    extracted_date, source_text = valid_dates[0]
                    logger.info(
                        f"Extracted date using datefinder: {extracted_date} from text: '{source_text}'"
                    )
                    return extracted_date
        except Exception as e:
            logger.debug(f"datefinder failed: {e}")

        logger.debug("No date found in content using any extraction method")
        return None

    def enrich_article_with_date(self, article: Article) -> Article:
        """Scrape article and update its date if missing (sync).

        Args:
            article: Article with potentially missing/guessed date

        Returns:
            Article with updated date from content
        """
        try:
            # Scrape content
            content = self.scrape_url(str(article.url))

            # Extract date
            extracted_date = self.extract_date_from_content(content)

            if extracted_date:
                logger.info(
                    f"Updated article date from {article.date} to {extracted_date} for: {article.title}"
                )
                # Create a new article with updated date
                return article.model_copy(update={"date": extracted_date})
            else:
                logger.warning(f"Could not extract date for article: {article.title}")
                return article

        except Exception as e:
            logger.error(f"Failed to enrich article {article.title}: {e}")
            return article

    async def enrich_article_with_date_async(self, article: Article) -> Article:
        """Scrape article and update its date if missing (async).

        Args:
            article: Article with potentially missing/guessed date

        Returns:
            Article with updated date from content
        """
        try:
            # Scrape content
            content = await self.scrape_url_async(str(article.url))

            # Extract date
            extracted_date = self.extract_date_from_content(content)

            if extracted_date:
                logger.info(
                    f"Updated article date from {article.date} to {extracted_date} for: {article.title}"
                )
                # Create a new article with updated date
                return article.model_copy(update={"date": extracted_date})
            else:
                logger.warning(f"Could not extract date for article: {article.title}")
                return article

        except Exception as e:
            logger.error(f"Failed to enrich article {article.title}: {e}")
            return article

    def enrich_classified_article_with_date(
        self, article: ClassifiedArticle
    ) -> ClassifiedArticle:
        """Scrape classified article and update its date if missing (sync).

        Args:
            article: ClassifiedArticle with potentially missing/guessed date

        Returns:
            ClassifiedArticle with updated date from content
        """
        try:
            # Scrape content
            content = self.scrape_url(str(article.url))

            # Extract date
            extracted_date = self.extract_date_from_content(content)

            if extracted_date:
                logger.info(
                    f"Updated article date from {article.date} to {extracted_date} for: {article.title}"
                )
                # Create a new article with updated date
                return article.model_copy(update={"date": extracted_date})
            else:
                logger.warning(f"Could not extract date for article: {article.title}")
                return article

        except Exception as e:
            logger.error(f"Failed to enrich article {article.title}: {e}")
            return article

    async def enrich_classified_article_with_date_async(
        self, article: ClassifiedArticle
    ) -> ClassifiedArticle:
        """Scrape classified article and update its date if missing (async).

        Args:
            article: ClassifiedArticle with potentially missing/guessed date

        Returns:
            ClassifiedArticle with updated date from content
        """
        try:
            # Scrape content
            content = await self.scrape_url_async(str(article.url))

            # Extract date
            extracted_date = self.extract_date_from_content(content)

            if extracted_date:
                logger.info(
                    f"Updated article date from {article.date} to {extracted_date} for: {article.title}"
                )
                # Create a new article with updated date
                return article.model_copy(update={"date": extracted_date})
            else:
                logger.warning(f"Could not extract date for article: {article.title}")
                return article

        except Exception as e:
            logger.error(f"Failed to enrich article {article.title}: {e}")
            return article

    async def enrich_articles_with_dates_async(
        self, articles: list[ClassifiedArticle]
    ) -> list[ClassifiedArticle]:
        """Enrich multiple articles with dates in parallel (async).

        Args:
            articles: List of ClassifiedArticles to enrich

        Returns:
            List of enriched ClassifiedArticles
        """
        tasks = [
            self.enrich_classified_article_with_date_async(article)
            for article in articles
        ]
        return await asyncio.gather(*tasks)


class ArticleSummary(BaseModel):
    """Detailed summary of a scraped article."""

    summary: str = Field(
        description="One paragraph (3-5 sentences) summarizing the key points and significance of the article"
    )


def summarize_article_content(article: ClassifiedArticle, scraped_content: str) -> str:
    """Generate a detailed 1-paragraph summary of the article content.

    Args:
        article: The classified article to summarize
        scraped_content: The full scraped content from Jina

    Returns:
        A detailed summary paragraph
    """
    client = _make_llm_client()
    model = _choose_model()  # Use free models from database

    prompt = f"""Summarize this AI news article in ONE paragraph (3-5 sentences). Focus on:
- What was announced/released/discovered
- Key technical details or capabilities
- Why it matters to the AI community

Article title: {article.title}
Article URL: {article.url}
Brief summary: {article.summary or "N/A"}

Full article content:
{scraped_content}

Write a clear, informative paragraph that captures the essence and significance of this article."""

    try:
        logger.info(
            f"Generating detailed summary for: {article.title} with model {model}"
        )
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert AI news summarizer."},
                {"role": "user", "content": prompt},
            ],
            response_format=ArticleSummary,
            temperature=0.4,
        )

        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("Failed to parse summary")

        return parsed.summary

    except Exception as e:
        logger.error(f"Failed to generate summary for {article.title}: {e}")
        return article.summary or "Summary unavailable"
