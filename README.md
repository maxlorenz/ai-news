# AI News Aggregator

Automated daily scraper that finds interesting AI news (model releases and agent research), classifies articles using free LLMs, stores them in MotherDuck, and sends Telegram summaries every morning.

## What It Does

- **Scrapes** 7 AI news sources (Hacker News, HuggingFace Papers, Apple ML, Google AI, Meta AI, MIT AI, Berkeley AI)
- **Classifies** articles using OpenRouter's free LLMs (focuses on new model releases and agent research)
- **Stores** interesting articles in MotherDuck (cloud DuckDB)
- **Sends** daily Telegram summaries at 8 AM Singapore time featuring:
  - Brief overview of all articles found
  - Top 3 articles (AI-selected) with detailed summaries scraped via Jina.ai

## Setup for GitHub Actions

### 1. OpenRouter API Key
- Sign up at [OpenRouter](https://openrouter.ai/)
- Go to [Keys page](https://openrouter.ai/keys)
- Create a new API key
- Add to GitHub Secrets as `OPENROUTER_API_KEY`

### 2. MotherDuck Token
- Sign up at [MotherDuck](https://motherduck.com/)
- Go to [Settings → Personal Access Tokens](https://app.motherduck.com/settings/tokens)
- Create a new token
- Add to GitHub Secrets as `MOTHERDUCK_TOKEN`

### 3. Telegram Bot
**Create Bot:**
- Message [@BotFather](https://t.me/botfather) on Telegram
- Send `/newbot` and follow instructions
- Copy the bot token (format: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)
- Add to GitHub Secrets as `TELEGRAM_BOT_TOKEN`

**Get Chat ID:**
- Create a Telegram group
- Add your bot to the group
- Send any message in the group
- Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
- Find the `"chat":{"id":-1234567890}` value (negative number for groups)
- Add to GitHub Secrets as `TELEGRAM_CHAT_ID`

**Enable Group Messages:**
- Message [@BotFather](https://t.me/botfather)
- Send `/mybots` → select your bot → Bot Settings → Group Privacy → Turn OFF

### 4. Add All Secrets to GitHub

Go to your repository → Settings → Secrets and variables → Actions → New repository secret

Add these 4 secrets:
- `OPENROUTER_API_KEY`
- `MOTHERDUCK_TOKEN`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Optional (for OpenRouter rankings):
- `OPENROUTER_HTTP_REFERER` - your site URL
- `OPENROUTER_X_TITLE` - your site title

## Running Locally

1. Install [uv](https://docs.astral.sh/uv/): `pip install uv`
2. Copy `.env.example` to `.env` and fill in credentials
3. Run: `uv run ai-news`

## Schedule

GitHub Actions runs automatically every day at **8:00 AM Singapore time (00:00 UTC)**.

You can also trigger manual runs from the Actions tab.
