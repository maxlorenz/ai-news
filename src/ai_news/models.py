from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, HttpUrl


class Source(str, Enum):
    HACKER_NEWS = "hacker_news"
    HUGGINGFACE_PAPERS = "huggingface_papers"


class Article(BaseModel):
    title: str
    url: HttpUrl
    source: Source
    date: datetime
    summary: Optional[str] = None


class ClassifiedArticle(Article):
    is_interesting: bool
    dedup_key: Optional[str] = None
