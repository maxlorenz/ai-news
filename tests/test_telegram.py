"""Tests for Telegram notification functions."""

import pytest
from pytest_httpx import HTTPXMock

from ai_news.settings import SETTINGS
from ai_news.telegram import send_telegram_message


@pytest.mark.unit
def test_send_telegram_message_success(httpx_mock: HTTPXMock, monkeypatch):
    """Test successful Telegram message sending."""
    monkeypatch.setattr(SETTINGS, "telegram_bot_token", "test_token")
    monkeypatch.setattr(SETTINGS, "telegram_chat_id", "-123456")

    httpx_mock.add_response(
        url="https://api.telegram.org/bottest_token/sendMessage",
        json={"ok": True, "result": {"message_id": 123}},
        status_code=200,
    )

    send_telegram_message("Test message")

    request = httpx_mock.get_request()
    assert request.method == "POST"
    # JSON has no space after colon when compact
    assert '"text":"Test message"' in request.content.decode()


@pytest.mark.unit
def test_send_telegram_message_splits_long_messages(httpx_mock: HTTPXMock, monkeypatch):
    """Test that long messages are split into chunks."""
    monkeypatch.setattr(SETTINGS, "telegram_bot_token", "test_token")
    monkeypatch.setattr(SETTINGS, "telegram_chat_id", "-123456")

    # Mock successful responses for 2 chunks
    httpx_mock.add_response(json={"ok": True}, status_code=200)
    httpx_mock.add_response(json={"ok": True}, status_code=200)

    # Create a message with newlines that will split into 2 chunks
    # Each line is 100 chars, so 50 lines = 5000 chars, will split into 2 chunks at 4000 char boundary
    lines = ["A" * 100 for _ in range(50)]
    long_message = "\n".join(lines)

    send_telegram_message(long_message)

    # Should have made 2 requests (split into chunks)
    requests = httpx_mock.get_requests()
    assert len(requests) == 2


@pytest.mark.unit
def test_send_telegram_message_no_credentials(monkeypatch):
    """Test that missing credentials skips sending."""
    monkeypatch.setattr(SETTINGS, "telegram_bot_token", None)
    monkeypatch.setattr(SETTINGS, "telegram_chat_id", None)

    # Should not raise, just log warning
    send_telegram_message("Test message")
