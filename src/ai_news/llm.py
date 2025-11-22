from __future__ import annotations

import json
import random
from typing import Iterable, List, Tuple

import backoff
from loguru import logger
from openai import OpenAI

from .models import Article, ClassifiedArticle
from .settings import FREE_OPENROUTER_MODELS, SETTINGS


def _make_client() -> OpenAI:
    return OpenAI(
        base_url=SETTINGS.openrouter_base_url,
        api_key=SETTINGS.openrouter_api_key,
    )


SYSTEM_PROMPT = """You are a strict filter for AI news.
You receive a list of short article candidates (title, url, source, and an overview snippet).

You must decide for each article:
- Is it about a NEW AI MODEL release (e.g. a new LLM/vision model or major version like Kimi K2)?
- OR is it about AGENT research (agent architectures, tools, benchmarks, or frameworks)?

If yes, mark it as interesting and provide a concise 1-3 sentence summary.
If not, mark it as not interesting.

Only use the given overview; DO NOT follow links.

Return strict JSON with the following schema:
{
  "results": [
    {
      "index": <integer index of the article in the input list>,
      "is_interesting": true | false,
      "summary": "short summary or empty string if not interesting",
      "dedup_key": "short normalized string combining model/agent name and key details"
    },
    ...
  ]
}
"""


def _build_user_prompt(articles: Iterable[Article]) -> str:
    payload = []
    for idx, art in enumerate(articles):
        payload.append(
            {
                "index": idx,
                "title": art.title,
                "url": str(art.url),
                "source": art.source.value,
                "date": art.date.isoformat(),
            }
        )
    return json.dumps({"articles": payload}, indent=2)


def _choose_model(exclude: List[str] | None = None) -> str:
    exclude = exclude or []
    candidates = [m for m in FREE_OPENROUTER_MODELS if m not in exclude]
    if not candidates:
        candidates = FREE_OPENROUTER_MODELS
    choice = random.choice(candidates)
    logger.debug("Using OpenRouter model: {}", choice)
    return choice


@backoff.on_exception(backoff.expo, Exception, max_tries=1)
def _call_openrouter(model: str, user_prompt: str) -> dict:
    client = _make_client()
    extra_headers = {}
    if SETTINGS.openrouter_referer:
        extra_headers["HTTP-Referer"] = SETTINGS.openrouter_referer
    if SETTINGS.openrouter_title:
        extra_headers["X-Title"] = SETTINGS.openrouter_title

    logger.info("Calling OpenRouter model {} for classification", model)

    response = client.chat.completions.create(
        extra_headers=extra_headers or None,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content or "{}"
    logger.debug("LLM raw response: {}", content)
    return json.loads(content)


def classify_articles(articles: List[Article]) -> List[ClassifiedArticle]:
    if not articles:
        return []

    # For simplicity, send them all in one batch; the lists are small.
    user_prompt = _build_user_prompt(articles)

    tried: List[str] = []
    last_exc: Exception | None = None
    for _ in range(len(FREE_OPENROUTER_MODELS)):
        model = _choose_model(tried)
        tried.append(model)
        try:
            data = _call_openrouter(model, user_prompt)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model {} failed: {}", model, exc)
            last_exc = exc
            continue
    else:
        raise RuntimeError("All OpenRouter models failed") from last_exc

    results = {int(r["index"]): r for r in data.get("results", [])}

    classified: List[ClassifiedArticle] = []
    for idx, art in enumerate(articles):
        r = results.get(idx) or {}
        is_interesting = bool(r.get("is_interesting", False))
        summary = (r.get("summary") or "").strip() or None
        dedup_key = (r.get("dedup_key") or "").strip() or None
        art_data = art.model_dump()
        art_data["summary"] = summary
        classified.append(
            ClassifiedArticle(
                **art_data,
                is_interesting=is_interesting,
                dedup_key=dedup_key,
            )
        )

    return classified


def detect_duplicates(
    new_articles: List[ClassifiedArticle],
    recent_articles: List[ClassifiedArticle],
) -> Tuple[List[ClassifiedArticle], List[ClassifiedArticle]]:
    """Return (unique_new, duplicates) based on URL, title, or dedup_key.

    - URL or exact title match is always considered a duplicate.
    - If dedup_key exists on both sides and matches case-insensitively, treat as duplicate.
    """

    url_set = {a.url for a in recent_articles}
    title_set = {a.title.strip().lower() for a in recent_articles}
    key_set = {a.dedup_key.strip().lower() for a in recent_articles if a.dedup_key}

    unique: List[ClassifiedArticle] = []
    dups: List[ClassifiedArticle] = []

    for art in new_articles:
        is_dup = False
        if art.url in url_set:
            is_dup = True
        elif art.title.strip().lower() in title_set:
            is_dup = True
        elif art.dedup_key and art.dedup_key.strip().lower() in key_set:
            is_dup = True

        if is_dup:
            dups.append(art)
        else:
            unique.append(art)

    logger.info("Duplicate detection: {} unique, {} duplicates", len(unique), len(dups))
    return unique, dups
