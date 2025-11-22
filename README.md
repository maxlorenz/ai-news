# AI News Scraper

Small uv-based Python app that scrapes AI-related news, classifies items using free OpenRouter-hosted LLMs, and stores interesting results in a local DuckDB database.

Sources:
- Hacker News front page (`https://news.ycombinator.com/`)
- Hugging Face AI papers for a given date (`https://huggingface.co/papers/date/YYYY-MM-DD`, using the current date)

The LLM filter only keeps:
- New AI model releases (e.g. new LLMs, new versions like Kimi K2)
- Agent research (agent architectures, benchmarks, frameworks, tooling)

Stored fields per article:
- `title`
- `url`
- `source`
- `date`
- `summary` (LLM-generated, based on overview only)
- `is_interesting`
- `dedup_key` (LLM-normalized key for duplicate detection)

## Project layout

- `src/ai_news/settings.py` – configuration and environment loading
- `src/ai_news/models.py` – shared Pydantic models and enums
- `src/ai_news/db.py` – DuckDB connection and helpers
- `src/ai_news/sources.py` – site-specific scraping utilities
- `src/ai_news/llm.py` – OpenRouter client, classification, and duplicate detection
- `src/ai_news/main.py` – orchestration logic
- `src/ai_news/__init__.py` – CLI entrypoint used by `ai-news` script

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed (`pip install uv` or via your package manager)
- Internet access so the app can reach Hacker News, Hugging Face, and OpenRouter

## Setup

1. Change into the project directory:

```bash
cd ai_news
```

2. Inspect or edit the `.env` file. It is created with the requested OpenRouter key by default:

```bash
cat .env
```

You can adjust:

- `OPENROUTER_API_KEY` – your personal key (recommended)
- `OPENROUTER_HTTP_REFERER` – optional site URL for OpenRouter rankings
- `OPENROUTER_X_TITLE` – optional site title for OpenRouter rankings
- `DUCKDB_PATH` – optional override for the DuckDB database file

3. Ensure dependencies are installed (already done when scaffolding, but safe to repeat):

```bash
uv sync
```

This will create a `.venv` and install all dependencies.

## Running the app

To run the scraper + classifier once:

```bash
uv run ai-news
```

This will:

1. Load configuration from `.env`.
2. Fetch candidate articles from Hacker News and Hugging Face (for today's date).
3. Call a randomly chosen free OpenRouter model (with automatic retries and fallbacks) to:
   - Decide whether each article is interesting (model releases or agent research only).
   - Generate a short summary and a `dedup_key` for each article.
4. Load articles from the past 48 hours from DuckDB.
5. De-duplicate new items against the recent ones using URL, title, and `dedup_key`.
6. Upsert interesting, non-duplicate articles into DuckDB.
7. Log a short snapshot of what is stored.

Log output is written both to the console and to `ai_news.log` in the project root.

## DuckDB database

By default the database file is created at:

```text
ai_news/ai_news.duckdb
```

You can inspect it using the DuckDB CLI:

```bash
uv run python -m duckdb ai_news.duckdb
```

Then, inside DuckDB:

```sql
SELECT * FROM articles ORDER BY date DESC LIMIT 20;
```

## Customizing

- To change the database path, set `DUCKDB_PATH` in `.env`.
- To change or pin the OpenRouter models used for classification, edit `FREE_OPENROUTER_MODELS` in `src/ai_news/settings.py`.

## Notes

- The app only uses the high-level metadata (titles and links from the listing pages). It does **not** crawl article bodies in depth.
- Duplicate detection uses a combination of URL, normalized title, and LLM-generated `dedup_key`, checked against articles stored in the last 48 hours.
