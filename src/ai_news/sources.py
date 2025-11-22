from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import HttpUrl

from .models import Article, Source

HN_URL = "https://news.ycombinator.com/"
HF_PAPERS_BASE = "https://huggingface.co/papers"
APPLE_ML_URL = "https://machinelearning.apple.com/research"
GOOGLE_AI_URL = "https://ai.google/research/"
META_AI_URL = "https://ai.meta.com/blog/"
MIT_AI_URL = "https://news.mit.edu/topic/artificial-intelligence2"
BERKELEY_AI_URL = "https://bair.berkeley.edu/blog/"


def parse_date_string(date_str: str) -> datetime | None:
    """Try to parse various date formats from blog pages."""
    if not date_str:
        return None

    # Clean up the string
    date_str = date_str.strip()

    # Common formats to try
    formats = [
        "%B %d, %Y",  # November 22, 2025
        "%b %d, %Y",  # Nov 22, 2025
        "%Y-%m-%d",  # 2025-11-22
        "%d %B %Y",  # 22 November 2025
        "%d %b %Y",  # 22 Nov 2025
        "%m/%d/%Y",  # 11/22/2025
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Try regex extraction for common patterns
    # Match "Nov 22, 2025" or "November 22, 2025"
    match = re.search(r"([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str)
    if match:
        try:
            return datetime.strptime(
                f"{match.group(1)} {match.group(2)}, {match.group(3)}", "%B %d, %Y"
            )
        except ValueError:
            try:
                return datetime.strptime(
                    f"{match.group(1)} {match.group(2)}, {match.group(3)}", "%b %d, %Y"
                )
            except ValueError:
                pass

    return None


async def fetch_hacker_news(client: httpx.AsyncClient) -> list[Article]:
    """Fetch articles from Hacker News front page.

    HN doesn't show dates on the front page, so we use current time.
    """
    logger.info("Fetching Hacker News front page")
    resp = await client.get(HN_URL, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[Article] = []

    for row in soup.select("tr.athing"):
        title_link = row.select_one("span.titleline a")
        if not title_link:
            continue
        title = title_link.get_text(strip=True)
        url = title_link["href"]

        try:
            art = Article(
                title=title,
                url=HttpUrl(str(url)),
                source=Source.HACKER_NEWS,
                date=datetime.now(UTC),  # HN doesn't show dates on front page
            )
            articles.append(art)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping HN item due to validation error: {}", exc)

    logger.info("Fetched {} candidate articles from HN", len(articles))
    return articles


async def fetch_huggingface_papers(
    client: httpx.AsyncClient,
    date: datetime,
) -> list[Article]:
    """Fetch papers from Hugging Face.

    Uses the papers page which shows recent papers.
    """
    url = HF_PAPERS_BASE
    logger.info("Fetching Hugging Face papers")

    resp = await client.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[Article] = []

    # Try to find paper cards
    for card in soup.select("article, .paper-card, div[class*='paper']")[:20]:
        # Find title
        title_el = card.select_one("h3, h2, h4, a[href*='/papers/']")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        if len(title) < 10:
            continue

        # Find link
        link_el = card.select_one("a[href*='/papers/']") or title_el
        if not link_el or link_el.name != "a":
            continue

        href = link_el.get("href")
        if not href:
            continue

        href_str = str(href)
        if href_str.startswith("/"):
            full_url = "https://huggingface.co" + href_str
        else:
            full_url = href_str

        # Try to extract date from the card
        date_el = card.select_one("time, .date, [class*='date']")
        article_date = date
        if date_el:
            date_text_raw = date_el.get("datetime") or date_el.get_text(strip=True)
            date_text = str(date_text_raw) if date_text_raw else ""
            parsed = parse_date_string(date_text)
            if parsed:
                article_date = parsed

        try:
            art = Article(
                title=title,
                url=HttpUrl(full_url),
                source=Source.HUGGINGFACE_PAPERS,
                date=article_date,
            )
            articles.append(art)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping HF item due to validation error: {}", exc)

    logger.info("Fetched {} candidate papers from Hugging Face", len(articles))
    return articles


async def fetch_apple_ml(client: httpx.AsyncClient) -> list[Article]:
    """Fetch articles from Apple Machine Learning Research.

    Dates are typically on the listing page or require fetching individual articles.
    """
    logger.info("Fetching Apple ML Research")
    try:
        resp = await client.get(APPLE_ML_URL)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Apple ML: {}", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[Article] = []

    for card in soup.select(
        "article, .research-item, .article-card, a[href*='/research/']"
    )[:20]:
        title_el = card.select_one("h2, h3, h4, .title") or card
        link_el = card if card.name == "a" else card.select_one("a")

        if not title_el or not link_el:
            continue

        title = title_el.get_text(strip=True)
        href = link_el.get("href")
        if not href or len(title) < 10:
            continue

        url_str = (
            str(href)
            if str(href).startswith("http")
            else f"https://machinelearning.apple.com{href}"
        )

        # Try to extract date from the card
        date_el = card.select_one("time, .date, [class*='date']")
        article_date = datetime.now(UTC)
        if date_el:
            date_text_raw = date_el.get("datetime") or date_el.get_text(strip=True)
            date_text = str(date_text_raw) if date_text_raw else ""
            parsed = parse_date_string(date_text)
            if parsed:
                article_date = parsed

        try:
            articles.append(
                Article(
                    title=title,
                    url=HttpUrl(url_str),
                    source=Source.APPLE_ML,
                    date=article_date,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping Apple ML item: {}", exc)

    logger.info("Fetched {} articles from Apple ML", len(articles))
    return articles[:10]


async def fetch_google_ai(client: httpx.AsyncClient) -> list[Article]:
    """Fetch articles from Google AI Research."""
    logger.info("Fetching Google AI Research")
    try:
        resp = await client.get(GOOGLE_AI_URL)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Google AI: {}", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[Article] = []

    for link in soup.select("a[href*='/research/']")[:20]:
        href = link.get("href")
        if not href:
            continue

        title_el = link.select_one("h2, h3, h4, .title") or link
        title = title_el.get_text(strip=True)

        if not title or len(title) < 10:
            continue

        url_str = (
            str(href) if str(href).startswith("http") else f"https://ai.google{href}"
        )

        # Try to find date near the link
        parent = link.find_parent("article") or link.find_parent("div")
        date_el = parent.select_one("time, .date, [class*='date']") if parent else None
        article_date = datetime.now(UTC)
        if date_el:
            date_text_raw = date_el.get("datetime") or date_el.get_text(strip=True)
            date_text = str(date_text_raw) if date_text_raw else ""
            parsed = parse_date_string(date_text)
            if parsed:
                article_date = parsed

        try:
            articles.append(
                Article(
                    title=title,
                    url=HttpUrl(url_str),
                    source=Source.GOOGLE_AI,
                    date=article_date,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping Google AI item: {}", exc)

    # Deduplicate
    seen_urls = set()
    unique = []
    for art in articles:
        if art.url not in seen_urls:
            seen_urls.add(art.url)
            unique.append(art)

    logger.info("Fetched {} articles from Google AI", len(unique))
    return unique[:10]


async def fetch_mit_ai(client: httpx.AsyncClient) -> list[Article]:
    """Fetch articles from MIT AI News."""
    logger.info("Fetching MIT AI News")
    try:
        resp = await client.get(MIT_AI_URL)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch MIT AI: {}", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[Article] = []

    for article in soup.select("article, .term-page--news-article")[:10]:
        title_el = article.select_one("h3 a, h2 a, .term-page--news-article--title a")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        href = title_el.get("href")
        if not href:
            continue

        url_str = (
            str(href) if str(href).startswith("http") else f"https://news.mit.edu{href}"
        )

        # Try to extract date
        date_el = article.select_one("time, .date, [class*='date']")
        article_date = datetime.now(UTC)
        if date_el:
            date_text_raw = date_el.get("datetime") or date_el.get_text(strip=True)
            date_text = str(date_text_raw) if date_text_raw else ""
            parsed = parse_date_string(date_text)
            if parsed:
                article_date = parsed

        try:
            articles.append(
                Article(
                    title=title,
                    url=HttpUrl(url_str),
                    source=Source.MIT_AI,
                    date=article_date,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping MIT AI item: {}", exc)

    logger.info("Fetched {} articles from MIT AI", len(articles))
    return articles


async def fetch_meta_ai(client: httpx.AsyncClient) -> list[Article]:
    """Fetch articles from Meta AI Blog."""
    logger.info("Fetching Meta AI Blog")
    try:
        resp = await client.get(META_AI_URL)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Meta AI: {}", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[Article] = []

    for link in soup.select("a[href*='/blog/']")[:20]:
        href = link.get("href")
        if not href or href == "/blog/":
            continue

        title_el = link.select_one("h2, h3, h4, .title") or link
        title = title_el.get_text(strip=True)

        if not title or len(title) < 10:
            continue

        url_str = (
            str(href) if str(href).startswith("http") else f"https://ai.meta.com{href}"
        )

        # For Meta, dates appear in text nodes near the article
        # Try to find date in the parent container's full text
        article_date = datetime.now(UTC)
        parent = link.find_parent("div") or link.find_parent("article")
        if parent:
            # Get all text from parent and search for date pattern
            parent_text = parent.get_text()
            # Match patterns like "Nov 21, 2025" or "November 21, 2025"
            date_match = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December|"
                r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(20\d{2})",
                parent_text,
            )
            if date_match:
                date_str = f"{date_match.group(1)} {date_match.group(2)}, {date_match.group(3)}"
                parsed = parse_date_string(date_str)
                if parsed:
                    article_date = parsed
                    logger.debug(
                        "Parsed Meta AI date: {} -> {}", date_str, article_date
                    )

        try:
            articles.append(
                Article(
                    title=title,
                    url=HttpUrl(url_str),
                    source=Source.META_AI,
                    date=article_date,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping Meta AI item: {}", exc)

    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for art in articles:
        if art.url not in seen_urls:
            seen_urls.add(art.url)
            unique_articles.append(art)

    logger.info("Fetched {} articles from Meta AI", len(unique_articles))
    return unique_articles[:10]


async def fetch_berkeley_ai(client: httpx.AsyncClient) -> list[Article]:
    """Fetch articles from Berkeley AI Research Blog."""
    logger.info("Fetching Berkeley AI Research Blog")
    try:
        resp = await client.get(BERKELEY_AI_URL)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Berkeley AI: {}", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    articles: list[Article] = []

    for post in soup.select(".post")[:10]:
        title_el = post.select_one("h1.post-title a, h2.post-title a")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        href = title_el.get("href")
        if not href:
            continue

        url_str = (
            str(href)
            if str(href).startswith("http")
            else f"https://bair.berkeley.edu{href}"
        )

        # Try to extract date from post - Berkeley uses span.post-meta for dates
        # Look for all span.post-meta elements and find the one with a date
        date_el = None
        for meta_span in post.select("span.post-meta"):
            text = meta_span.get_text(strip=True)
            # Skip spans that contain links (those are author names)
            if not meta_span.select("a") and text:
                date_el = meta_span
                break

        article_date = datetime.now(UTC)
        if date_el:
            date_text = date_el.get_text(strip=True)
            parsed = parse_date_string(date_text)
            if parsed:
                article_date = parsed
                logger.debug("Parsed Berkeley date: {} -> {}", date_text, article_date)

        try:
            articles.append(
                Article(
                    title=title,
                    url=HttpUrl(url_str),
                    source=Source.BERKELEY_AI,
                    date=article_date,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping Berkeley AI item: {}", exc)

    logger.info("Fetched {} articles from Berkeley AI", len(articles))
    return articles


async def fetch_all_sources(current_date: datetime) -> list[Article]:
    """Fetch articles from all sources and combine them."""
    async with httpx.AsyncClient(
        headers={"User-Agent": "ai-news-bot/0.1"}, follow_redirects=True, timeout=30
    ) as client:
        hn_articles = await fetch_hacker_news(client)
        hf_articles = await fetch_huggingface_papers(client, current_date)
        apple_articles = await fetch_apple_ml(client)
        meta_articles = await fetch_meta_ai(client)
        google_articles = await fetch_google_ai(client)
        mit_articles = await fetch_mit_ai(client)
        berkeley_articles = await fetch_berkeley_ai(client)

    combined: list[Article] = [
        *hn_articles,
        *hf_articles,
        *apple_articles,
        *meta_articles,
        *google_articles,
        *mit_articles,
        *berkeley_articles,
    ]
    logger.info("Total raw articles fetched: {}", len(combined))
    return combined
