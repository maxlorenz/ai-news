from __future__ import annotations

import random
from collections.abc import Iterable

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from .db import get_available_openrouter_models
from .models import Article, ClassifiedArticle
from .settings import SETTINGS


def _make_client() -> OpenAI:
    return OpenAI(
        base_url=SETTINGS.openrouter_base_url,
        api_key=SETTINGS.openrouter_api_key,
    )


SYSTEM_PROMPT = """You are a strict filter for AI news.
You receive a list of article candidates with title, url, source, date, and optionally an overview snippet.

You must decide for each article based on the TITLE (and overview if available):
- Is it about a NEW AI MODEL release (e.g. a new LLM, vision model, or major version like GPT-5, Gemini 2.5, Claude 4, Llama 4, Qwen 3, DeepSeek, etc.)?
- OR is it about AGENT/AGENTIC AI research or tools (agent architectures, agent frameworks, multi-agent systems, tool use, agent benchmarks)?

Examples of INTERESTING articles:
- "OpenAI releases GPT-5" 
- "Anthropic announces Claude 4"
- "Google launches Gemini 2.5 Flash"
- "Meta releases Llama 4"
- "New multi-agent framework for..."
- "Agent benchmarking suite..."
- "Tool-augmented agents..."

Examples of NOT interesting:
- General tech news
- Hardware/infrastructure (unless specifically about AI training/inference)
- Software engineering tools (unless agent-related)
- Business/startup news
- General programming articles

If yes to either criterion, mark it as interesting and provide a concise 1-2 sentence summary.
If not, mark it as not interesting (leave summary and dedup_key empty).

IMPORTANT: Be somewhat lenient - if a title mentions AI models, LLMs, agents, or related terms, mark it as interesting even if you're not 100% certain.
"""


class ArticleInput(BaseModel):
    index: int
    title: str
    url: str
    source: str
    date: str
    overview: str = ""  # Add overview field


class ArticlesInput(BaseModel):
    articles: list[ArticleInput]


class ClassificationResult(BaseModel):
    index: int = Field(description="Integer index of the article in the input list")
    is_interesting: bool = Field(description="Whether the article is interesting")
    summary: str = Field(
        default="", description="Short summary or empty string if not interesting"
    )
    dedup_key: str = Field(
        default="",
        description="Short normalized string combining model/agent name and key details",
    )


class ClassificationResponse(BaseModel):
    results: list[ClassificationResult]


def _build_user_input(articles: Iterable[Article]) -> ArticlesInput:
    payload = []
    for idx, art in enumerate(articles):
        # Use the summary field if available (populated by sources)
        overview = art.summary or ""
        payload.append(
            ArticleInput(
                index=idx,
                title=art.title,
                url=str(art.url),
                source=art.source.value,
                date=art.date.isoformat(),
                overview=overview,
            )
        )
    return ArticlesInput(articles=payload)


def _choose_model(exclude: list[str] | None = None) -> str:
    exclude = exclude or []

    # Load models from database
    db_models = get_available_openrouter_models()
    if not db_models:
        raise RuntimeError(
            "No OpenRouter models available in database. "
            "Please run the OpenRouter model fetch first."
        )

    candidates = [m for m in db_models if m not in exclude]
    if not candidates:
        candidates = db_models

    choice = random.choice(candidates)
    logger.debug("Using OpenRouter model from DB: {}", choice)
    return choice


def _call_openrouter(model: str, user_input: ArticlesInput) -> ClassificationResponse:
    client = _make_client()
    extra_headers = {}
    if SETTINGS.openrouter_referer:
        extra_headers["HTTP-Referer"] = SETTINGS.openrouter_referer
    if SETTINGS.openrouter_title:
        extra_headers["X-Title"] = SETTINGS.openrouter_title

    logger.info("Calling OpenRouter model {} for classification", model)

    # Log what we're sending to the LLM
    input_json = user_input.model_dump_json(indent=2)
    logger.debug(
        "Sending to LLM: {}",
        input_json[:500] + "..." if len(input_json) > 500 else input_json,
    )

    response = client.beta.chat.completions.parse(
        extra_headers=extra_headers or None,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": input_json,
            },
        ],
        response_format=ClassificationResponse,
        temperature=0.2,
    )

    parsed = response.choices[0].message.parsed
    if not parsed:
        raise ValueError("Failed to parse response from LLM")
    logger.debug("LLM parsed response: {}", parsed)
    return parsed


def classify_articles(articles: list[Article]) -> list[ClassifiedArticle]:
    if not articles:
        return []

    # For simplicity, send them all in one batch; the lists are small.
    user_input = _build_user_input(articles)

    # Get available models from DB
    available_models = get_available_openrouter_models()
    if not available_models:
        raise RuntimeError(
            "No OpenRouter models available in database. "
            "Please run the OpenRouter model fetch first."
        )

    tried: list[str] = []
    last_exc: Exception | None = None
    for _ in range(len(available_models)):
        model = _choose_model(tried)
        tried.append(model)
        try:
            data = _call_openrouter(model, user_input)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model {} failed: {}", model, exc)
            last_exc = exc
            continue
    else:
        raise RuntimeError("All OpenRouter models failed") from last_exc

    results = {r.index: r for r in data.results}

    classified: list[ClassifiedArticle] = []
    for idx, art in enumerate(articles):
        r = results.get(idx)
        if r:
            is_interesting = r.is_interesting
            summary = r.summary.strip() or None
            dedup_key = r.dedup_key.strip() or None
        else:
            is_interesting = False
            summary = None
            dedup_key = None

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
    new_articles: list[ClassifiedArticle],
    recent_articles: list[ClassifiedArticle],
) -> tuple[list[ClassifiedArticle], list[ClassifiedArticle]]:
    """Return (unique_new, duplicates) based on URL, title, or dedup_key.

    - URL or exact title match is always considered a duplicate.
    - If dedup_key exists on both sides and matches case-insensitively, treat as duplicate.
    """

    url_set = {a.url for a in recent_articles}
    title_set = {a.title.strip().lower() for a in recent_articles}
    key_set = {a.dedup_key.strip().lower() for a in recent_articles if a.dedup_key}

    unique: list[ClassifiedArticle] = []
    dups: list[ClassifiedArticle] = []

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
