from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, HttpUrl


class Source(str, Enum):
    HACKER_NEWS = "hacker_news"
    HUGGINGFACE_PAPERS = "huggingface_papers"
    APPLE_ML = "apple_ml"
    GOOGLE_AI = "google_ai"
    META_AI = "meta_ai"
    MIT_AI = "mit_ai"
    BERKELEY_AI = "berkeley_ai"


class Article(BaseModel):
    title: str
    url: HttpUrl
    source: Source
    date: datetime
    summary: str | None = None


class ClassifiedArticle(Article):
    is_interesting: bool
    dedup_key: str | None = None
