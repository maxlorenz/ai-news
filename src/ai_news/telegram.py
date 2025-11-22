from __future__ import annotations

import httpx
from loguru import logger

from .models import ClassifiedArticle
from .reports import DailyReport, generate_daily_report
from .settings import SETTINGS


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


def send_daily_summary(articles: list[ClassifiedArticle]) -> DailyReport:
    """Generate and send the daily summary via Telegram.

    Args:
        articles: List of interesting articles to include in report

    Returns:
        DailyReport object with all report metadata
    """
    logger.info("Generating daily summary for {} articles", len(articles))

    try:
        # Generate report
        report = generate_daily_report(articles)

        # Send via Telegram
        send_telegram_message(report.report_content)
        logger.info("Daily summary sent successfully")

        return report
    except Exception as e:
        logger.error(f"Failed to send daily summary: {e}")
        raise
