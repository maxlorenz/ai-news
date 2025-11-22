from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger
from pydantic import BaseModel, Field

from .llm import _choose_model
from .llm import _make_client as _make_llm_client
from .models import ClassifiedArticle
from .scraper import JinaScraper, summarize_article_content


class Top3Selection(BaseModel):
    """AI-selected top 3 articles with reasoning."""

    article_indices: list[int] = Field(
        description="List of 3 article indices (0-based) representing the most important/interesting articles",
        min_length=1,
        max_length=3,
    )
    reasoning: str = Field(
        description="Brief explanation of why these 3 articles were chosen"
    )


class DailyReport(BaseModel):
    """Daily AI news report with metadata."""

    report_date: datetime
    article_count: int
    report_content: str  # Markdown formatted
    top_3_article_urls: list[str]
    selection_reasoning: str | None = None


def select_top_3_articles(
    articles: list[ClassifiedArticle],
) -> tuple[list[ClassifiedArticle], str | None]:
    """Use AI to select the top 3 most important articles.

    Args:
        articles: List of classified articles to choose from

    Returns:
        Tuple of (top 3 articles, reasoning for selection)
    """
    if len(articles) <= 3:
        return articles, None

    client = _make_llm_client()
    model = _choose_model()  # Use free models from database

    # Build prompt with article info
    article_list = []
    for idx, art in enumerate(articles):
        article_list.append(
            f"{idx}. {art.title}\n   URL: {art.url}\n   Summary: {art.summary or 'No summary'}"
        )

    articles_text = "\n\n".join(article_list)

    prompt = f"""You are analyzing a list of AI news articles. Select the TOP 3 most important and impactful articles.

Consider:
- Major model releases (GPT, Claude, Gemini, Llama, etc.) are highest priority
- Novel agent architectures or frameworks
- Significant research breakthroughs
- Industry-wide impact

Articles:
{articles_text}

Select exactly 3 articles by their index numbers (0-based)."""

    try:
        logger.info(
            "Using AI to select top 3 articles from {} candidates with model {}",
            len(articles),
            model,
        )
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI news curator selecting the most important articles.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=Top3Selection,
            temperature=0.3,
        )

        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("Failed to parse AI response")

        logger.info(
            "AI selected articles: {} - Reasoning: {}",
            parsed.article_indices,
            parsed.reasoning,
        )

        # Return selected articles
        top_3 = [articles[idx] for idx in parsed.article_indices if idx < len(articles)]
        return top_3[:3], parsed.reasoning

    except Exception as e:
        logger.error(f"Failed to select top 3 with AI: {e}, falling back to first 3")
        return articles[:3], None


def generate_daily_report(articles: list[ClassifiedArticle]) -> DailyReport:
    """Generate a daily report with top 3 article deep dives.

    Args:
        articles: List of interesting articles from today

    Returns:
        DailyReport object with all metadata and formatted content
    """
    report_date = datetime.now(UTC)

    if not articles:
        return DailyReport(
            report_date=report_date,
            article_count=0,
            report_content="📭 No interesting AI news today!",
            top_3_article_urls=[],
            selection_reasoning=None,
        )

    # Header
    message_parts = [
        "🤖 *Daily AI News Summary*",
        f"📊 Found {len(articles)} interesting article{'s' if len(articles) != 1 else ''} today\n",
    ]

    # Brief overview of all articles
    message_parts.append("📰 *All Articles:*")
    for idx, art in enumerate(articles, 1):
        message_parts.append(f"{idx}. [{art.title}]({art.url})")
        if art.summary:
            message_parts.append(f"   _{art.summary}_\n")

    message_parts.append("\n" + "=" * 50 + "\n")

    # Select and detail top 3
    logger.info("Selecting top 3 articles for detailed summaries...")
    top_3, reasoning = select_top_3_articles(articles)

    message_parts.append("🌟 *Top 3 Deep Dives:*\n")

    scraper = JinaScraper()

    for idx, art in enumerate(top_3, 1):
        message_parts.append(f"*{idx}. {art.title}*")
        message_parts.append(f"🔗 {art.url}\n")

        # Scrape once, then retry LLM summarization separately
        try:
            scraped_content = scraper.scrape_url(str(art.url))
        except Exception as e:
            logger.error(f"Failed to scrape article {art.title}: {e}")
            message_parts.append(f"📝 {art.summary or 'Summary unavailable'}\n")
            continue

        # Try to summarize with retries (without re-scraping)
        detailed_summary = art.summary or "Summary unavailable"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                detailed_summary = summarize_article_content(art, scraped_content)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to summarize {art.title} (attempt {attempt + 1}/{max_retries}): {e}, retrying..."
                    )
                else:
                    logger.error(
                        f"Failed to summarize {art.title} after {max_retries} attempts: {e}"
                    )

        message_parts.append(f"📝 {detailed_summary}\n")

    report_content = "\n".join(message_parts)

    return DailyReport(
        report_date=report_date,
        article_count=len(articles),
        report_content=report_content,
        top_3_article_urls=[str(art.url) for art in top_3],
        selection_reasoning=reasoning,
    )
