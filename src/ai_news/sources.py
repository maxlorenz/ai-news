from __future__ import annotations

from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import HttpUrl

from .models import Article, Source


HN_URL = "https://news.ycombinator.com/"
HF_PAPERS_BASE = "https://huggingface.co/papers/date/{date}"


async def fetch_hacker_news(client: httpx.AsyncClient) -> list[Article]:
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
                date=datetime.utcnow(),
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
    day_str = date.strftime("%Y-%m-%d")
    url = HF_PAPERS_BASE.format(date=day_str)
    logger.info("Fetching Hugging Face papers for {}", day_str)

    resp = await client.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    articles: list[Article] = []

    # The page structure may evolve; we target generic paper cards.
    for card in soup.select("a.paper-card, a.block"):
        title_el = card.select_one("h2, h3")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        href_attr = card.get("href") or ""
        href = str(href_attr)
        if href.startswith("/"):
            full_url = "https://huggingface.co" + href
        else:
            full_url = href

        if not full_url:
            continue

        try:
            art = Article(
                title=title,
                url=HttpUrl(full_url),
                source=Source.HUGGINGFACE_PAPERS,
                date=date,
            )
            articles.append(art)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping HF item due to validation error: {}", exc)

    logger.info("Fetched {} candidate papers from Hugging Face", len(articles))
    return articles


async def fetch_all_sources(current_date: datetime) -> list[Article]:
    async with httpx.AsyncClient(headers={"User-Agent": "ai-news-bot/0.1"}) as client:
        hn_articles = await fetch_hacker_news(client)
        hf_articles = await fetch_huggingface_papers(client, current_date)

    combined: list[Article] = [*hn_articles, *hf_articles]
    logger.info("Total raw articles fetched: {}", len(combined))
    return combined
