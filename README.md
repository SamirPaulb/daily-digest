# Daily Digest

Automated daily briefing — markets, news, AI, startups, investing, careers. Generated every day at 4:30 AM IST via GitHub Actions.

**Live**: [samirpaulb.github.io/daily/](https://samirpaulb.github.io/daily/)

## How It Works

```
4:30 AM IST daily (or manual trigger)
    ↓
Fetch market data (Finnhub → Alpha Vantage → Yahoo Finance → Twelve Data → Massive)
    ↓
Fetch top gainers/losers
  US:    Yahoo screener (mcap ≥ $10B) → Alpha Vantage
  India: NSE equity-stockIndices (Nifty 50)
    ↓
Fetch news (Tavily → NewsAPI → GNews → NYTimes → Currents → Mediastack → Finnhub → Exa → RSS)
    ↓
AI generates digest sections (13+ providers with fallback chain)
    ↓
Script injects real market numbers (never AI-generated)
    ↓
Appends Further Reading links from RSS
    ↓
Converts to HTML → commits to repo
    ↓
Main blog fetches HTML via raw.githubusercontent.com
```

## AI Provider Fallback Chain

| Level | Providers | Type |
|-------|-----------|------|
| 1 | Gemini, OpenAI, OpenRouter/Perplexity, Z.AI, DeepSeek, xAI, Claude | Search-capable AI |
| 1.5 | Direct assembly | No AI — builds from pre-fetched data |
| 2 | Gemini, OpenAI, DeepSeek, Z.AI, Groq, xAI, Mistral, OpenRouter, Fireworks, Moonshot, MiniMax, GitHub Models | Standard AI |
| 2.5 | Ollama (qwen2.5:7b Q4) | Local model on runner |
| 3 | Data template | Markets + headlines only |
| 4 | Blank template | Always succeeds |
| Workflow | GitHub AI Inference → Vercel AI Gateway → Ollama install | Action-level fallbacks |

## Market Data Sources

| Data | Primary | Fallback chain |
|------|---------|----------------|
| Index prices | Finnhub | Alpha Vantage → Yahoo Finance → Twelve Data → Massive |
| US movers | Yahoo screener (mcap ≥ $10B) | Alpha Vantage TOP_GAINERS_LOSERS |
| India movers | NSE equity-stockIndices | Skipped if unavailable (geo-blocked from non-Indian IPs) |

## Secrets (GitHub → Settings → Secrets → Actions)

### Required (at least one AI + one news)
- `FINNHUB_API_KEY` — market data
- `OPENROUTER_API_KEY` — AI (Perplexity search)

### Recommended
- `GEMINI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `FIREWORKS_API_KEY` — AI redundancy
- `TAVILY_API_KEY`, `NEWS_API_KEY`, `GNEWS_API_KEY` — news sources
- `ALPHAVANTAGE_API_KEY` — market data + US movers backup

### Optional (more redundancy)
- `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `ZAI_API_KEY`
- `MOONSHOT_AI_API_KEY`, `MINIMAX_API_KEY`
- `NYTIMES_API_KEY`, `CURRENTS_API_KEY`, `MEDIASTACK_API_KEY`, `EXA_API_KEY`
- `TWELVEDATA_API_KEY` — market data backup (800 credits/day free)
- `MASSIVE_API_KEY` — market data (crypto/forex only on free plan)
- `VERCEL_AI_GATEWAY_API_KEY` — workflow fallback

### Always available (no setup needed)
- `GITHUB_TOKEN` — GitHub Models + AI Inference action

## Variables (GitHub → Settings → Variables → Actions)

All optional. Override model names when a provider deprecates one — no code change needed.

```
GEMINI_MODEL, OPENAI_MODEL, OPENAI_SEARCH_MODEL, DEEPSEEK_MODEL,
GROQ_MODEL, MISTRAL_MODEL, XAI_MODEL, ZAI_MODEL, FIREWORKS_MODEL,
MOONSHOT_MODEL, MINIMAX_MODEL, OPENROUTER_SEARCH_MODEL, OPENROUTER_FREE_MODEL,
GITHUB_MODEL, TWELVEDATA_BASE_URL
```

## Local Development

```bash
cp .env.example .env
# Fill in API keys

pip install -r scripts/requirements.txt
python scripts/generate_digest.py          # generate today's digest
python scripts/generate_digest.py --test   # test all providers
```

## Manual Trigger

Actions → "Generate Daily Digest" → Run workflow → `test_mode: false`

Set `test_mode: true` to validate all API keys without generating output.
