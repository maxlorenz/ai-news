from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import polars as pl
from loguru import logger

from .models import ClassifiedArticle


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _month_path(dt: datetime) -> Path:
    return DATA_DIR / f"articles_{dt:%Y_%m}.parquet"


def _articles_to_df(articles: Iterable[ClassifiedArticle]) -> pl.DataFrame:
    rows = [
        {
            "url": str(a.url),
            "title": a.title,
            "source": a.source.value,
            "date": a.date,
            "summary": a.summary,
            "is_interesting": a.is_interesting,
            "dedup_key": a.dedup_key,
        }
        for a in articles
    ]
    if not rows:
        return pl.DataFrame(
            schema={
                "url": pl.Utf8,
                "title": pl.Utf8,
                "source": pl.Utf8,
                "date": pl.Datetime,
                "summary": pl.Utf8,
                "is_interesting": pl.Boolean,
                "dedup_key": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def _df_to_articles(df: pl.DataFrame) -> List[ClassifiedArticle]:
    articles: List[ClassifiedArticle] = []
    for row in df.to_dicts():
        articles.append(
            ClassifiedArticle(
                url=row["url"],
                title=row["title"],
                source=row["source"],
                date=row["date"],
                summary=row.get("summary"),
                is_interesting=row.get("is_interesting"),
                dedup_key=row.get("dedup_key"),
            )
        )
    return articles


def load_all_articles() -> List[ClassifiedArticle]:
    if not DATA_DIR.exists():
        return []

    frames: List[pl.DataFrame] = []
    for path in sorted(DATA_DIR.glob("articles_*.parquet")):
        try:
            frames.append(pl.read_parquet(path))
        except FileNotFoundError:
            continue

    if not frames:
        return []

    df = pl.concat(frames, how="vertical_relaxed")
    return _df_to_articles(df)


def load_recent_articles(hours: int = 48) -> List[ClassifiedArticle]:
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    if not DATA_DIR.exists():
        return []

    frames: List[pl.DataFrame] = []
    for path in sorted(DATA_DIR.glob("articles_*.parquet")):
        try:
            df = pl.read_parquet(path)
        except FileNotFoundError:
            continue
        if "date" not in df.columns:
            continue
        df_filtered = df.filter(pl.col("date") >= cutoff)
        if df_filtered.height:
            frames.append(df_filtered)

    if not frames:
        return []

    df = pl.concat(frames, how="vertical_relaxed")
    return _df_to_articles(df)


def upsert_articles(articles: Iterable[ClassifiedArticle]) -> int:
    arts = list(articles)
    if not arts:
        logger.info("No articles to upsert into Parquet storage")
        return 0

    # We only store interesting articles here.
    by_month: dict[tuple[int, int], list[ClassifiedArticle]] = {}
    for a in arts:
        key = (a.date.year, a.date.month)
        by_month.setdefault(key, []).append(a)

    total_written = 0

    for (year, month), month_articles in by_month.items():
        month_dt = datetime(year=year, month=month, day=1)
        path = _month_path(month_dt)
        logger.info("Writing {} articles to {}", len(month_articles), path)

        new_df = _articles_to_df(month_articles)

        if path.exists():
            existing_df = pl.read_parquet(path)
            # De-duplicate on URL (primary key semantics)
            combined = pl.concat([existing_df, new_df], how="vertical_relaxed")
            combined = combined.unique(subset=["url"], keep="last")
            combined.write_parquet(path)
            total_written += len(new_df)
        else:
            new_df.write_parquet(path)
            total_written += len(new_df)

    return total_written
