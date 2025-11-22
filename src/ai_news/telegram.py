from __future__ import annotations

import httpx
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from .models import ClassifiedArticle
from .settings import SETTINGS
from .llm import _choose_model, _make_client as _make_llm_client


def send_telegram_message(message: str) -> None:
    """Send a message to the configured Telegram chat.

    If message is too long (>4096 chars), it will be split into multiple messages.
    """
    if not SETTINGS.telegram_bot_token or not SETTINGS.telegram_chat_id:
        logger.warning("Telegram credentials not configured, skipping notification")
        return

    url = f"https://api.telegram.org/bot{SETTINGS.telegram_bot_token}/sendMessage"

    # Mask token for logging (show first 10 and last 5 chars)
    token_masked = (
        f"{SETTINGS.telegram_bot_token[:10]}...{SETTINGS.telegram_bot_token[-5:]}"
        if len(SETTINGS.telegram_bot_token) > 15
        else "***"
    )
    logger.debug(
        f"Telegram config - Bot token: {token_masked}, Chat ID: {SETTINGS.telegram_chat_id}"
    )

    # Telegram message limit is 4096 characters
    MAX_MESSAGE_LENGTH = 4000  # Leave some buffer

    # Split message if too long
    if len(message) > MAX_MESSAGE_LENGTH:
        logger.warning(
            f"Message too long ({len(message)} chars), splitting into chunks"
        )
        chunks = []
        current_chunk = ""

        for line in message.split("\n"):
            # If adding this line would exceed the limit, start a new chunk
            if len(current_chunk) + len(line) + 1 > MAX_MESSAGE_LENGTH:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += ("\n" if current_chunk else "") + line

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Split message into {len(chunks)} chunks")

        # Send each chunk
        for idx, chunk in enumerate(chunks, 1):
            logger.debug(f"Sending chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
            _send_single_message(url, chunk, token_masked)
    else:
        logger.debug(f"Sending message of length {len(message)} chars")
        _send_single_message(url, message, token_masked)


def _send_single_message(url: str, message: str, token_masked: str) -> None:
    """Send a single Telegram message (helper function)."""
    try:
        response = httpx.post(
            url,
            json={
                "chat_id": SETTINGS.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": False,
            },
            timeout=30.0,
        )

        # Log response details before raising
        if response.status_code != 200:
            logger.error(
                f"Telegram API error - Status: {response.status_code}, Response: {response.text}"
            )
            logger.error(
                f"Request payload - chat_id: {SETTINGS.telegram_chat_id}, message_length: {len(message)}, parse_mode: Markdown"
            )

        response.raise_for_status()
        logger.info("Telegram message sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        logger.error(f"Bot token (masked): {token_masked}")
        logger.error(f"Chat ID: {SETTINGS.telegram_chat_id}")
        logger.error(f"Message preview (first 200 chars): {message[:200]}")
        raise


def scrape_with_jina(url: str) -> str:
    """Scrape article content using Jina.ai Reader API."""
    jina_url = f"https://r.jina.ai/{url}"

    try:
        logger.info(f"Scraping article with Jina.ai: {url}")
        response = httpx.get(jina_url, timeout=30.0)
        response.raise_for_status()
        content = response.text

        # Limit content to avoid token limits (keep first 8000 chars ~2000 tokens)
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated...]"

        logger.info(f"Successfully scraped {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Failed to scrape article {url}: {e}")
        return f"Failed to scrape content: {str(e)}"


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


def select_top_3_articles(articles: list[ClassifiedArticle]) -> list[ClassifiedArticle]:
    """Use AI to select the top 3 most important articles."""
    if len(articles) <= 3:
        return articles

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
        return top_3[:3]  # Ensure max 3

    except Exception as e:
        logger.error(f"Failed to select top 3 with AI: {e}, falling back to first 3")
        return articles[:3]


class ArticleSummary(BaseModel):
    """Detailed summary of a scraped article."""

    summary: str = Field(
        description="One paragraph (3-5 sentences) summarizing the key points and significance of the article"
    )


def summarize_article_content(article: ClassifiedArticle, scraped_content: str) -> str:
    """Generate a detailed 1-paragraph summary of the article content."""
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


def create_daily_summary(articles: list[ClassifiedArticle]) -> str:
    """Create the full daily summary message with top 3 detailed articles."""
    if not articles:
        return "📭 No interesting AI news today!"

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
    top_3 = select_top_3_articles(articles)

    message_parts.append("🌟 *Top 3 Deep Dives:*\n")

    for idx, art in enumerate(top_3, 1):
        message_parts.append(f"*{idx}. {art.title}*")
        message_parts.append(f"🔗 {art.url}\n")

        # Scrape and summarize
        try:
            scraped_content = scrape_with_jina(str(art.url))
            detailed_summary = summarize_article_content(art, scraped_content)
            message_parts.append(f"📝 {detailed_summary}\n")
        except Exception as e:
            logger.error(f"Failed to process article {art.title}: {e}")
            message_parts.append(f"📝 {art.summary or 'Summary unavailable'}\n")

    return "\n".join(message_parts)


def send_daily_summary(articles: list[ClassifiedArticle]) -> None:
    """Generate and send the daily summary via Telegram."""
    logger.info("Generating daily summary for {} articles", len(articles))

    try:
        summary = create_daily_summary(articles)
        send_telegram_message(summary)
        logger.info("Daily summary sent successfully")
    except Exception as e:
        logger.error(f"Failed to send daily summary: {e}")
        raise
