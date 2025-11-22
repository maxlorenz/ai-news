from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger


JINA_BASE = "https://r.jina.ai/https://"


async def fetch_via_jina(url: str, client: Optional[httpx.AsyncClient] = None) -> str:
    """Fetch page content via Jina Reader proxy.

    Example of equivalent curl call:
        curl "https://r.jina.ai/https://www.example.com"

    We only need the rendered HTML/markdown that Jina returns; parsing is handled
    by the caller.
    """

    target = url.lstrip("https://").lstrip("http://")
    jina_url = JINA_BASE + target

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(headers={"User-Agent": "ai-news-bot/0.1"})

    assert client is not None

    try:
        logger.debug("Fetching via Jina Reader: {}", jina_url)
        resp = await client.get(jina_url, timeout=30)
        resp.raise_for_status()
        return resp.text
    finally:
        if own_client:
            await client.aclose()
