# Daily Digest

Automated daily briefing — markets, news, AI, startups, investing, careers. Generated every day at 4:30 AM IST via GitHub Actions.

**Live:** [samirpaulb.github.io/daily/](https://samirpaulb.github.io/daily/)

## How It Works

```
4:30 AM IST daily (GitHub Actions cron, or manual trigger)
    ↓
Fetch market data — 5-provider fallback:
  Finnhub → Alpha Vantage → Yahoo Finance → Twelve Data → Massive
    ↓
Fetch top gainers/losers:
  US:    Yahoo screener (mcap ≥ $10B) → Alpha Vantage
  India: NSE equity-stockIndices (Nifty 50)
    ↓
Fetch news — 20+ sources in parallel, 60s timeout:
  Tavily, RSS (25+ feeds), NewsAPI, GNews, DDG, WebSearchAPI,
  NYTimes, Currents, Mediastack, Finnhub, Exa, NewsData,
  WorldNewsAPI, NewsCatcher, Mojeek
    ↓
Deduplicate across all sections
    ↓
AI generates digest (15-25 items/section → picks best 7-10)
    ↓
Script injects real market numbers (never AI-generated)
    ↓
Appends Further Reading links from RSS
    ↓
Converts Markdown → HTML, commits to repo
    ↓
Main blog fetches HTML via raw.githubusercontent.com
```

## Project Structure

```
daily-digest/
├── scripts/
│   ├── generate_digest.py    # Main generator (~3200 lines)
│   └── requirements.txt      # Python dependencies
├── digests/                   # Generated HTML digests (YYYY-MM-DD.html)
├── index.html                 # Web viewer with dark mode
├── manifest.json              # Digest metadata index
├── .github/workflows/
│   └── generate.yml           # Daily cron + manual trigger
├── .env.example               # API key template
└── docs/                      # Reserved for documentation
```

## News Sources

All sources run in parallel. AI receives the combined, deduplicated pool and picks the best stories.

| Source | Type | Sections | Key Required |
|--------|------|----------|--------------|
| Tavily | AI search | Global, India, Tech, Finance | Yes |
| RSS feeds | 25+ feeds (BBC, Reuters, TOI, TechCrunch, etc.) | Global, India, Tech | No |
| NewsAPI | Structured news | Global, India, Tech | Yes |
| GNews | Structured news | Global, India, Tech | Yes |
| DDG News | DuckDuckGo news (`ddgs` package) | Global, India, Tech | No |
| WebSearchAPI.ai | Google-powered search | Global | Yes |
| NYTimes | Top Stories / Most Popular | Global, Tech | Yes |
| Currents | Global news | Global, India, Tech | Yes |
| Mediastack | Multi-source | Global, India, Tech | Yes |
| NewsData | Multi-source | Global, India, Tech | Yes |
| WorldNewsAPI | Multi-source | Global, India, Tech | Yes |
| NewsCatcher | Multi-source | Global, India, Tech | Yes |
| Finnhub News | Market news | Global | Yes |
| Exa | Neural search | Global, India, Tech | Yes |
| Mojeek | Independent search | All (last resort) | No |

## AI Provider Fallback Chain

Six levels of fallback ensure a digest is always generated, even if every API is down.

| Level | Providers | Strategy |
|-------|-----------|----------|
| 1 | Gemini, OpenAI (`gpt-4.1`), OpenRouter (`gpt-oss-120b:free`), Z.AI, DeepSeek, xAI, Claude | Search-capable AI |
| 1.5 | Direct assembly | No AI — builds from pre-fetched data |
| 2 | Gemini, OpenAI (`gpt-4.1-mini`), DeepSeek, Z.AI, Groq, xAI, Mistral, OpenRouter (`nemotron-3-super:free`), Fireworks, Moonshot, MiniMax, GitHub Models | Standard AI |
| 2.5 | Ollama (`qwen2.5:7b` Q4) | Local model on GitHub runner |
| 3 | Data template | Markets + headlines only, no AI |
| 4 | Blank template | Always succeeds |

Workflow-level fallbacks: GitHub AI Inference → Vercel AI Gateway → Ollama install.

## Market Data Sources

| Data | Primary | Fallback Chain |
|------|---------|----------------|
| Index prices | Finnhub | Alpha Vantage → Yahoo Finance → Twelve Data → Massive |
| US movers | Yahoo screener (mcap ≥ $10B) | Alpha Vantage TOP_GAINERS_LOSERS |
| India movers | NSE equity-stockIndices | Skipped if unavailable (geo-blocked outside India) |

## GitHub Models (Free Tier)

Uses `GH_MODELS_PAT` (PAT with `models:read` scope) for the GitHub Models endpoint at `models.github.ai`. Tries multiple models in order:

1. `openai/gpt-4.1-mini` (primary)
2. `deepseek/DeepSeek-V3-0324`
3. `meta/Llama-4-Scout-17B-16E-Instruct`
4. `microsoft/Phi-4-reasoning` (free long-term, Microsoft-hosted)

Configurable via `GITHUB_MODEL` and `GITHUB_MODEL_FALLBACKS` variables.

## Setup

### Secrets (GitHub → Settings → Secrets → Actions)

**Required** (at least one AI + one news source):
- `FINNHUB_API_KEY` — market data
- `OPENROUTER_API_KEY` — AI (free models available)

**Recommended:**
- `GH_MODELS_PAT` — GitHub Models PAT (`models:read` scope, free tier)
- `GEMINI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `FIREWORKS_API_KEY` — AI redundancy
- `TAVILY_API_KEY`, `NEWS_API_KEY`, `GNEWS_API_KEY` — news sources
- `WEBSEARCHAPIAI_API_KEY` — Google-powered search (1000 free credits/month)
- `ALPHAVANTAGE_API_KEY` — market data + US movers backup

**Optional** (more redundancy):
- `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `ZAI_API_KEY`
- `MOONSHOT_AI_API_KEY`, `MINIMAX_API_KEY`
- `NYTIMES_API_KEY`, `CURRENTS_API_KEY`, `MEDIASTACK_API_KEY`, `EXA_API_KEY`
- `NEWSDATAIO_API_KEY`, `WORLDNEWSAPI_API_KEY`, `NEWSCATCHERAPI_API_KEY`
- `TWELVEDATA_API_KEY` — market data (800 credits/day free)
- `MASSIVE_API_KEY` — market data (crypto/forex only on free plan)
- `VERCEL_AI_GATEWAY_API_KEY` — workflow fallback

**Always available** (no setup needed):
- `GITHUB_TOKEN` — fallback for GitHub Models if `GH_MODELS_PAT` not set

### Variables (GitHub → Settings → Variables → Actions)

All optional. Override model names when a provider deprecates one — no code change needed.

```
GEMINI_MODEL, OPENAI_MODEL, OPENAI_SEARCH_MODEL, DEEPSEEK_MODEL,
GROQ_MODEL, MISTRAL_MODEL, XAI_MODEL, ZAI_MODEL, FIREWORKS_MODEL,
MOONSHOT_MODEL, MINIMAX_MODEL, OPENROUTER_SEARCH_MODEL, OPENROUTER_FREE_MODEL,
GITHUB_MODEL, GITHUB_MODEL_FALLBACKS, TWELVEDATA_BASE_URL
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

## Disclaimer

The daily digest content is AI-generated and may contain factual errors. Market data is sourced from third-party APIs and may be delayed or inaccurate. This is not financial advice. Always verify information from authoritative sources.

## License

MIT
