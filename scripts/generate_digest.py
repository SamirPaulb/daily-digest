"""
generate_digest.py — Daily Digest with 6-level fallback
========================================================
Designed to run unchanged for 10+ years.

ARCHITECTURE — pre-fetch first, then AI:
  All data is fetched ONCE at startup (Yahoo Finance + HN + Tavily/Exa/DDG/Mojeek news).
  Every AI provider receives the same rich context so even standard models produce
  quality digests — no model needs to search independently.

FALLBACK LEVELS (tried in order):
  Level 1    AI + pre-fetched context + optional extra search
               (Claude with web_search, OpenAI search-preview, Gemini grounding,
                OpenRouter Perplexity — each also gets the pre-fetched news data)
  Level 1.5  Direct assembly — no LLM, pre-fetched data only
               (builds digest straight from Tavily/Exa/DDG/Mojeek results)
  Level 2    Standard AI + pre-fetched context (no extra search)
               (Claude, OpenAI, Gemini, OpenRouter free model, GitHub Models)
  Level 2.5  Local Ollama model (gemma2:2b-instruct-q8_0 — distilled from 9B, near-lossless Q8)
               — installed by the workflow only when all cloud APIs have failed
  Level 3    Data-only template  (stdlib; same search chain for news sections)
  Level 4    Blank template      (pure stdlib, zero deps — ALWAYS SUCCEEDS)

MODEL NAMES are read from env vars so you can update them via GitHub
Variables (Settings → Variables → Actions) without touching this file.

CONFIGURATION — set as GitHub Variables (not Secrets):
  CLAUDE_MODEL             default: claude-haiku-4-5-20251001
  CLAUDE_SEARCH_TOOL       default: web_search_20250305
  OPENAI_SEARCH_MODEL      default: gpt-4o-mini-search-preview
  OPENAI_MODEL             default: gpt-4o-mini
  GEMINI_MODEL             default: gemini-2.0-flash
  OPENROUTER_SEARCH_MODEL  default: perplexity/llama-3.1-sonar-small-128k-online
  OPENROUTER_FREE_MODEL    default: google/gemini-2.0-flash-exp:free
  GITHUB_MODEL             default: gpt-4o-mini  (models.inference.ai.azure.com)

API KEYS — set as GitHub Secrets:
  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
  TAVILY_API_KEY   — tavily.com (used at Level 1.5 and Level 3, news + finance)
  EXA_API_KEY      — exa.ai (fallback when Tavily section returns empty)
  NEWS_API_KEY     — newsapi.org (primary per-section news fallback)
  GNEWS_API_KEY    — gnews.io (secondary per-section news fallback)
  CURRENTS_API_KEY — currentsapi.services (tertiary per-section news fallback)
  FINNHUB_API_KEY  — finnhub.io (Finnhub market news + quote fallback for indices)
  (set any subset — only providers with keys are tried)

NOTE: GITHUB_TOKEN is auto-injected in Actions and used by the Level 2
  urllib fallback (GitHub Models). No extra secrets needed for that level.
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import feedparser
import markdown as md_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Date / paths
# ──────────────────────────────────────────────────────────────────────────────

IST  = timezone(timedelta(hours=5, minutes=30))
_now = datetime.now(IST)

DATE_ISO   = _now.strftime("%Y-%m-%d")           # 2026-05-07
DATE_HUMAN = _now.strftime("%B %d, %Y")          # May 07, 2026
DATE_FRONT = _now.strftime("%Y-%m-%dT07:00:00+05:30")

OUTPUT_DIR  = Path("digests")
OUTPUT_FILE = OUTPUT_DIR / f"{DATE_ISO}.html"
MANIFEST    = Path("manifest.json")

# ──────────────────────────────────────────────────────────────────────────────
# Config — model/tool names from env vars with safe defaults
# When a model is deprecated: go to GitHub → Settings → Variables → Actions
# and update the variable. No code change needed.
# ──────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str) -> str:
    """Return env var value, falling back to default if unset or empty."""
    return os.environ.get(key) or default

# ──────────────────────────────────────────────────────────────────────────────
# All configurable variables — change via GitHub Variables, never touch code.
#
# MODEL NAMES: Update when a provider deprecates a model.
# BASE URLS:   Update if a provider changes their API endpoint.
# API KEYS:    Set as GitHub Secrets (read from env at runtime).
#
# To update: GitHub → Settings → Variables → Actions → edit the variable.
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # ── AI Provider Models ────────────────────────────────────────────────
    # Level 1 (search-capable, ranked by quality):
    "GEMINI_MODEL":            _env("GEMINI_MODEL",            "gemini-2.0-flash"),
    "OPENAI_SEARCH_MODEL":     _env("OPENAI_SEARCH_MODEL",     "gpt-4o-mini-search-preview-2025-03-11"),
    "OPENROUTER_SEARCH_MODEL": _env("OPENROUTER_SEARCH_MODEL", "perplexity/sonar"),
    "DEEPSEEK_MODEL":          _env("DEEPSEEK_MODEL",          "deepseek-v4-flash"),
    "XAI_MODEL":               _env("XAI_MODEL",               "grok-3-mini-fast"),
    "CLAUDE_MODEL":            _env("CLAUDE_MODEL",            "claude-haiku-4-5-20251001"),
    "CLAUDE_SEARCH_TOOL":      _env("CLAUDE_SEARCH_TOOL",      "web_search_20250305"),

    # Level 2 (standard, ranked by quality — same models, no search):
    "OPENAI_MODEL":            _env("OPENAI_MODEL",            "gpt-4o-mini"),
    "OPENROUTER_FREE_MODEL":   _env("OPENROUTER_FREE_MODEL",   "google/gemini-2.0-flash:free"),
    "GROQ_MODEL":              _env("GROQ_MODEL",              "llama-3.3-70b-versatile"),
    "MISTRAL_MODEL":           _env("MISTRAL_MODEL",           "mistral-small-latest"),
    "FIREWORKS_MODEL":         _env("FIREWORKS_MODEL",          "accounts/fireworks/models/deepseek-v3p1"),
    "MOONSHOT_MODEL":          _env("MOONSHOT_MODEL",           "kimi-k2.6"),
    "MINIMAX_MODEL":           _env("MINIMAX_MODEL",            "MiniMax-M2.5"),
    "ZAI_MODEL":               _env("ZAI_MODEL",               "glm-4.5"),
    "GITHUB_MODEL":            _env("GITHUB_MODEL",            "gpt-4o-mini"),

    # ── AI Provider Base URLs ─────────────────────────────────────────────
    "OPENAI_BASE_URL":         _env("OPENAI_BASE_URL",         "https://api.openai.com/v1"),
    "OPENROUTER_BASE_URL":     _env("OPENROUTER_BASE_URL",     "https://openrouter.ai/api/v1"),
    "DEEPSEEK_BASE_URL":       _env("DEEPSEEK_BASE_URL",       "https://api.deepseek.com"),
    "XAI_BASE_URL":            _env("XAI_BASE_URL",            "https://api.x.ai/v1"),
    "GROQ_BASE_URL":           _env("GROQ_BASE_URL",           "https://api.groq.com/openai/v1"),
    "MISTRAL_BASE_URL":        _env("MISTRAL_BASE_URL",        "https://api.mistral.ai/v1"),
    "FIREWORKS_BASE_URL":      _env("FIREWORKS_BASE_URL",      "https://api.fireworks.ai/inference/v1"),
    "MOONSHOT_BASE_URL":       _env("MOONSHOT_BASE_URL",       "https://api.moonshot.ai/v1"),
    "MINIMAX_BASE_URL":        _env("MINIMAX_BASE_URL",        "https://api.minimax.io/v1"),
    "ZAI_BASE_URL":            _env("ZAI_BASE_URL",            "https://api.z.ai/api/paas/v4"),
    "GITHUB_MODELS_BASE_URL":  _env("GITHUB_MODELS_BASE_URL",  "https://models.inference.ai.azure.com"),
}


def _openai_compatible_call(
    api_key_env: str, base_url_key: str, model_cfg_key: str, prompt: str,
    timeout: float = 90.0, max_tokens: int = 2048,
    extra_body: Optional[dict] = None,
) -> str:
    """
    Generic caller for any OpenAI-compatible API.
    Used by: DeepSeek, Mistral, Groq, xAI, Fireworks, Moonshot, MiniMax, OpenRouter.
    Pure try/except — never raises, returns empty string on any failure.
    """
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=CFG[base_url_key], timeout=timeout)
        kwargs: dict[str, Any] = {
            "model": CFG[model_cfg_key],
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception:
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# Logging — timestamps make it easy to spot slow steps in CI logs
# ──────────────────────────────────────────────────────────────────────────────

def _log(tag: str, msg: str) -> None:
    ts = datetime.now(IST).strftime("%H:%M:%S")
    print(f"{ts} [{tag:<6}] {msg}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Retry helper — exponential back-off for transient network failures
# ──────────────────────────────────────────────────────────────────────────────

def _retry(fn: Callable, attempts: int = 3, base_delay: float = 1.5) -> Any:
    """Retry fn up to `attempts` times with exponential back-off on any exception."""
    last_exc: BaseException = RuntimeError("no attempts made")
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                delay = base_delay * (2 ** attempt)
                _log("WARN", f"  retry {attempt + 1}/{attempts} in {delay:.1f}s — {exc}")
                time.sleep(delay)
    raise last_exc

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# HN meta-post prefixes — community posts, not real tech/news items
_HN_META_PREFIXES = ("ask hn:", "show hn:", "tell hn:", "launch hn:")

# Placeholder markers that indicate an AI returned template content, not real data.
# _validate() rejects any output containing these strings.
_PLACEHOLDER_PATTERNS = ("[DRAFT", "[verify]", "[Headline]", "[price]")

# Substrings that identify generic news-site homepage titles scraped from DDG/Mojeek.
# These are meta-titles like "Latest News Today, Top Headlines & Live Updates — News24",
# not actual article headlines. Filtered out before storing results.
_JUNK_TITLE_FRAGMENTS = (
    "latest news today",
    "breaking news",
    "top headlines",
    "live updates",
    "live news",
    "top news stories",
    "news today",
    "today's news",
    "today news",
    "latest updates",
    "all news",
    "news live",
    "samachar",         # Hindi: "news" — generic aggregator titles
    "ताजा समाचार",      # Hindi: "latest news"
)

# RSS feeds per section — tried in order when Exa has no key / returns nothing.
# Uses free, well-known feeds; no API key required.  feedparser handles RSS 2.0,
# Atom, encoding issues, and malformed XML entities automatically.
# Multiple feeds per section provide redundancy if one URL changes.
# First URL that returns non-empty results wins; the rest serve as fallback.
_RSS_SECTION_FEEDS: dict[str, list[str]] = {
    "global": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",           # BBC World (stable since ~2000)
        "https://feeds.reuters.com/reuters/worldNews",            # Reuters World
        "https://feeds.reuters.com/reuters/topNews",              # Reuters Top
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # NYT Top Stories
        "https://www.theguardian.com/world/rss",                  # The Guardian World (free)
        "https://feeds.washingtonpost.com/rss/world",             # Washington Post World
        "https://feeds.bloomberg.com/markets/news.rss",           # Bloomberg Markets
        "https://feeds.bloomberg.com/politics/news.rss",          # Bloomberg Politics
        "https://www.cnbc.com/id/100003114/device/rss/rss.html", # CNBC Top News
        "https://www.thehindu.com/feeder/default.rss",            # The Hindu (international perspective)
        "https://www.ft.com/rss/home/international",              # Financial Times International
    ],
    "india": [
        "https://timesofindia.indiatimes.com/rssfeedsdefault.cms",    # Times of India top stories
        "https://www.thehindu.com/feeder/default.rss",                 # The Hindu (main feed)
        "https://www.thehindu.com/feedly/s1/india/feedly.rss",        # The Hindu India section
        "https://economictimes.indiatimes.com/rssfeed/1977021501.cms", # Economic Times
        "https://feeds.feedburner.com/NdtvProfit-LatestNews",          # NDTV Profit
        "https://www.livemint.com/rss/news",                           # Livemint
        "https://www.moneycontrol.com/rss/lateststories.xml",          # MoneyControl
    ],
    "tech": [
        "https://feeds.bloomberg.com/technology/news.rss",        # Bloomberg Technology
        "https://feeds.arstechnica.com/arstechnica/index",        # Ars Technica
        "https://techcrunch.com/feed/",                           # TechCrunch
        "https://www.theverge.com/rss/index.xml",                 # The Verge (Atom)
        "https://www.wired.com/feed/rss",                         # Wired
        "https://www.technologyreview.com/feed/",                 # MIT Technology Review
        "https://www.forbes.com/innovation/feed",                 # Forbes Innovation
    ],
}

# RSS feeds for "Further Reading" section — diverse, high-quality sources
_FURTHER_READING_FEEDS: list[str] = [
    "https://fortune.com/feed/",                                  # Fortune
    "https://www.ft.com/?format=rss",                             # Financial Times
    "https://www.forbes.com/innovation/feed",                     # Forbes Innovation
    "https://www.reddit.com/r/selfimprovement.rss",               # Reddit Self-Improvement
    "https://techcrunch.com/feed/",                               # TechCrunch
    "https://feeds.bloomberg.com/markets/news.rss",               # Bloomberg Markets
    "https://hbr.org/feed",                                       # Harvard Business Review
]

# ──────────────────────────────────────────────────────────────────────────────
# Output format template (embedded here so no external file dependency)
# ──────────────────────────────────────────────────────────────────────────────

_EXPECTED_FORMAT = f"""\
Output ONLY markdown — no preamble, no explanation, no code fences.
DO NOT generate a ## Markets section — it is injected separately by the script.

STRUCTURE RULES:
1. Start with YAML front matter (title, date, summary)
2. Include 5-7 sections (## heading + bullet points) — NO Markets section
3. Sections separated by --- (horizontal rule)
4. Each section: 2-4 bullet points, format: - **Bold headline** — brief detail.
5. Be smart about what's newsworthy TODAY — skip sections with nothing interesting
6. End with a ## Further Reading section (2-4 links)

PICK 5-7 of these based on what's most interesting/relevant today:
- ## Global News — geopolitics, world events, breaking news
- ## India — Indian politics, economy, business, sports
- ## AI & Tech — AI breakthroughs, product launches, tech policy, developer news
- ## Startups & Funding — funding rounds, acquisitions, new startups, sector trends
- ## Investing & Predictions — analyst calls, bank forecasts, stock/commodity outlook
- ## Career & Opportunities — hiring trends, hot skills, remote jobs, career moves
- ## Personal Finance — savings tips, rate changes, tax, insurance, budgeting
- ## Learning & Growth — one skill/course/book/resource worth exploring today
- ## Insight of the Day — one powerful tweet, quote, or non-obvious observation

FORMAT:

---
title: "Daily Digest — {DATE_HUMAN}"
date: {DATE_FRONT}
summary: "One punchy sentence covering 2-3 top stories"
---

## [Section Name]

- **Headline** — brief detail.
- **Headline** — brief detail.

---

(repeat for 5-7 sections)"""

def _prompt_with_rich_data(
    market: dict,
    hn: list,
    glob_news: list,
    india_news: list,
    tech_news: list,
    mkt_commentary: str = "",
    search_hint: bool = False,
) -> str:
    """
    Build a prompt pre-loaded with all pre-fetched real-time data.
    Market data is NOT passed to AI (handled separately by the script).
    search_hint=True  → for search-capable models (Claude, GPT-search, Gemini grounding,
                         Perplexity) — they can supplement the data with their own search.
    search_hint=False → for standard models and Ollama — data is self-contained.
    """
    hn_lines = "\n".join(f"  - {h}" for h in hn[:10]) or "  [fetch failed]"

    def _sec(items: list, label: str) -> str:
        if items:
            return "\n".join(f"  - {item}" for item in items[:8])
        hint = f"search for today's {label} stories" if search_hint else "use general knowledge"
        return f"  [no pre-fetched data — {hint}]"

    # Blend tech search results with HN headlines for the tech section
    tech_combined = tech_news[:5] + [f"**{h}**" for h in hn[:5]]

    supplement = (
        "\n\nYou have live web search — use it to fill sections like Startups & Funding, "
        "Investing & Predictions, Career & Opportunities, and Personal Finance with REAL, "
        "verifiable today's news. Only include items you can confirm via search. "
        "For Global News, India, and AI & Tech — prefer the pre-fetched data below."
        if search_hint else
        "\n\nFor sections without pre-fetched data: SKIP them rather than guessing. "
        "Only output sections you have real data for."
    )

    return f"""\
You are a smart personal daily briefing writer. Your reader is a software engineer \
based in India who actively invests globally (Indian mutual funds, US stocks, UCITS, \
global equities), follows AI/startups/tech, and wants to stay informed about career \
opportunities, market moves, and emerging sectors without missing anything important.
Today is {DATE_HUMAN}.

NOTE: The Markets section (prices, indices) is handled separately by the script — \
do NOT generate any market data or prices. Focus ONLY on news and insights.{supplement}

ACCURACY RULES (CRITICAL):
1. News sections (Global, India, AI & Tech): use ONLY headlines from the PRE-FETCHED data below. \
You may rephrase for brevity but NEVER invent a headline or event that isn't in the data.
2. For sections without pre-fetched data (Startups, Investing, Career, Personal Finance, Learning): \
ONLY include if you have web search results confirming it. If unsure, skip the section entirely. \
Never hallucinate company names, funding amounts, analyst names, or predictions.
3. If a section would have fewer than 2 real items, skip it — do not pad with invented content.
4. Always prefer fewer accurate items over more questionable ones.

GLOBAL NEWS (verified headlines):
{_sec(glob_news, 'global')}

INDIA NEWS (verified headlines):
{_sec(india_news, 'India')}

TECH / AI / STARTUPS / JOBS (verified headlines):
{_sec(tech_combined, 'tech/AI/startups/jobs')}

HACKER NEWS (developer community — real titles):
{hn_lines}

{_EXPECTED_FORMAT}"""

# ──────────────────────────────────────────────────────────────────────────────
# Output normalisation and validation
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED = ["## "]  # At least one section heading required from AI


def _normalize(text: str) -> str:
    """Strip code fences, leading preamble, and surrounding whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Strip markdown code fence wrapper (```markdown ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]           # remove opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]          # remove closing fence
        text = "\n".join(lines).strip()
    # Strip any preamble before the front-matter opening ---
    idx = text.find("---")
    if idx > 0:
        text = text[idx:]
    return text.strip()


def _validate(text: str) -> bool:
    """
    Return True if text looks like a valid, publishable digest.

    Rejects:
    - Short or structurally incomplete outputs
    - Outputs containing placeholder markers (template not filled in)
    - Outputs with fewer than 2 real headline bullets
    """
    text = _normalize(text)
    if len(text) < 300:
        return False
    if not text.startswith("---"):
        return False
    if not all(s in text for s in _REQUIRED):
        return False
    # Reject outputs that still contain placeholder markers —
    # these indicate the AI echoed the template rather than generating real content
    for placeholder in _PLACEHOLDER_PATTERNS:
        if placeholder in text:
            return False
    # Require at least 2 real headline bullets (- **...**)
    if text.count("- **") < 2:
        return False
    return True


def _clean(text: str) -> str:
    """Return normalised text guaranteed to end with a single newline."""
    return _normalize(text) + "\n"


# Map source identifiers → human-readable author label for front matter.
# Data-only and blank-template levels produce no AI author, so they are omitted.
_SOURCE_AUTHOR: dict[str, str] = {
    "claude":         "Claude",
    "claude+data":    "Claude",
    "openai":         "OpenAI",
    "openai+data":    "OpenAI",
    "gemini":         "Gemini",
    "gemini+data":    "Gemini",
    "openrouter":     "OpenRouter",
    "openrouter+data":"OpenRouter",
    "github-models":  "GitHub Models",
    "ollama":         "Local AI",
    # tavily-direct and data-only are data-derived, no AI author
}


def _inject_author(text: str, author: str) -> str:
    """
    Insert 'author: "Name"' into the YAML front matter block.
    No-op if author is blank or if 'author:' is already present.
    """
    if not author:
        return text
    # Front matter must start at position 0
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    fm = text[3:end]
    if "author:" in fm:
        return text
    return text[:end] + f'\nauthor: "{author}"' + text[end:]


def _dedup_news(
    glob_news: list, india_news: list, tech_news: list,
) -> tuple[list, list, list]:
    """
    Remove cross-section and within-section duplicate headlines.
    Uses the first 60 normalised characters as the dedup key.
    Processes sections in order (glob → india → tech), so earlier sections
    get priority when the same story appears in multiple sections.
    """
    seen: set[str] = set()

    def _key(item: str) -> str:
        # Strip markdown bold markers before comparing
        return item.replace("**", "").lower()[:60].strip()

    def _dedup(items: list) -> list:
        out = []
        for item in items:
            k = _key(item)
            if k and k not in seen:
                seen.add(k)
                out.append(item)
        return out

    return _dedup(glob_news), _dedup(india_news), _dedup(tech_news)

# ──────────────────────────────────────────────────────────────────────────────
# Data fetchers — stdlib urllib only, no packages required
# ──────────────────────────────────────────────────────────────────────────────

# Yahoo Finance has two equivalent query hosts; try both for resilience.
_YAHOO_ENDPOINTS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d",
    "https://query2.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d",
]


_NSE_HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_nse_nifty() -> Optional[dict]:
    """
    Official Nifty 50 daily change from NSE India.

    Primary:  Historical EOD API — queries last 10 calendar days to span
              weekends/holidays; takes the two most recent sessions to compute
              daily % change = (close_today - close_prev) / close_prev * 100.
    Fallback: Intraday API — returns lastPrice + pChange (% vs prev close)
              directly; useful when the EOD data isn't published yet.

    Returns {"price": "...", "change": "..."} or None on any error.
    """
    # ── Primary: historical EOD ───────────────────────────────────────────────
    try:
        to_date   = _now.strftime("%d-%m-%Y")
        from_date = (_now - timedelta(days=10)).strftime("%d-%m-%Y")
        url = (
            "https://www.nseindia.com/api/historicalOR/indicesHistory"
            f"?indexType=NIFTY%2050&from={from_date}&to={to_date}"
        )
        req = urllib.request.Request(url, headers=_NSE_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            records = (json.load(resp).get("data") or [])
        if records:
            close = float(records[-1].get("EOD_CLOSE_INDEX_VAL") or 0)
            if close:
                if len(records) >= 2:
                    prev  = float(records[-2].get("EOD_CLOSE_INDEX_VAL") or close)
                    perc  = ((close - prev) / prev * 100) if prev else 0.0
                else:
                    perc = 0.0
                _log("DATA", f"  Nifty 50 (NSE EOD): {close:,.2f} ({perc:+.2f}%)")
                return {"price": f"{close:,.2f}", "change": f"{perc:+.2f}%"}
    except Exception as exc:
        _log("WARN", f"  NSE historical failed: {exc}")

    # ── Fallback: intraday API ────────────────────────────────────────────────
    try:
        url = (
            "https://www.nseindia.com/api/equity-stockIndices"
            "?index=NIFTY%2050"
        )
        req = urllib.request.Request(url, headers=_NSE_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        for rec in (data.get("data") or []):
            if (rec.get("symbol") or "").upper() in ("NIFTY 50", "NIFTY50"):
                price = float(rec.get("lastPrice") or rec.get("last") or 0)
                perc  = float(rec.get("pChange") or rec.get("percentChange") or 0)
                _log("DATA", f"  Nifty 50 (NSE intraday): {price:,.2f} ({perc:+.2f}%)")
                return {"price": f"{price:,.2f}", "change": f"{perc:+.2f}%"}
    except Exception as exc:
        _log("WARN", f"  NSE intraday failed: {exc}")

    return None


def _fetch_bse_sensex() -> Optional[dict]:
    """
    Official Sensex data from BSE India's public REST API.
    Returns {"price": "...", "change": "..."} or None on any error.
    """
    try:
        url = (
            "https://api.bseindia.com/BseIndiaAPI/api/GetIndexData/w"
            "?indexnm=BSE%20SENSEX"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (digest-bot/1.0)",
                "Referer": "https://www.bseindia.com/",
                "Origin": "https://www.bseindia.com",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        # Response: {"Data": [{"IndexName": "BSE SENSEX", "CurrValue": "77123.45",
        #                       "PercChange": "0.31", ...}]}
        for rec in (data.get("Data") or []):
            idx = (rec.get("IndexName") or "").upper()
            if "SENSEX" in idx:
                price = float(rec.get("CurrValue") or 0)
                perc  = float(
                    str(rec.get("PercChange") or "0")
                    .replace("%", "").replace("+", "")
                )
                return {"price": f"{price:,.2f}", "change": f"{perc:+.2f}%"}
    except Exception as exc:
        _log("WARN", f"  BSE official failed: {exc}")
    return None


def _fetch_yahoo_quote(sym: str) -> Optional[dict]:
    """
    Fetch a single symbol from Yahoo Finance.
    Tries query1 first, falls back to query2.  Each host retried twice.
    Returns {"price": "...", "change": "..."} or None on all errors.
    """
    def _fetch_url(url: str) -> dict:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)

    sym_enc = urllib.parse.quote(sym)
    for url_tmpl in _YAHOO_ENDPOINTS:
        url = url_tmpl.format(sym=sym_enc)
        try:
            data = _retry(lambda u=url: _fetch_url(u), attempts=2, base_delay=2.0)
            meta  = data["chart"]["result"][0]["meta"]
            price = float(meta.get("regularMarketPrice") or 0)
            prev  = float(meta.get("chartPreviousClose") or price) or price
            chg   = ((price - prev) / prev * 100) if prev else 0.0
            return {"price": f"{price:,.2f}", "change": f"{chg:+.2f}%"}
        except Exception as exc:
            _log("WARN", f"  Yahoo {url_tmpl.split('/')[2]} ({sym}) failed: {exc}")
    return None


def _fetch_finnhub_quote(sym: str) -> Optional[dict]:
    """
    Fetch a quote from Finnhub (primary source for market data).
    Supports US stocks, major indices, forex, crypto.

    Returns {"price": "...", "change": "..."} or None on error/missing key.
    Ref: https://finnhub.io/docs/api/quote
    Response: {"c": current_price, "d": abs_change, "dp": pct_change, "pc": prev_close}
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return None
    try:
        url = (
            "https://finnhub.io/api/v1/quote"
            f"?symbol={urllib.parse.quote(sym)}&token={api_key}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        price = float(data.get("c") or 0)
        perc  = float(data.get("dp") or 0)
        if price:
            return {"price": f"{price:,.2f}", "change": f"{perc:+.2f}%"}
    except Exception as exc:
        _log("WARN", f"  Finnhub quote ({sym}) failed: {exc}")
    return None


def _fetch_alphavantage_quote(sym: str) -> Optional[dict]:
    """
    Fetch a quote from Alpha Vantage (secondary source).
    Free tier: 25 calls/day. Use as fallback after Finnhub.

    Supports US stocks/ETFs via GLOBAL_QUOTE.
    For forex (USDINR=X style): uses CURRENCY_EXCHANGE_RATE endpoint.
    Ref: https://www.alphavantage.co/documentation/

    Returns {"price": "...", "change": "..."} or None on error/missing key.
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        return None
    try:
        # Forex pairs (e.g., USDINR=X → from=USD, to=INR)
        if "=X" in sym:
            pair = sym.replace("=X", "")
            from_cur = pair[:3]
            to_cur = pair[3:]
            url = (
                "https://www.alphavantage.co/query"
                f"?function=CURRENCY_EXCHANGE_RATE"
                f"&from_currency={from_cur}&to_currency={to_cur}"
                f"&apikey={api_key}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.load(resp)
            rate_data = data.get("Realtime Currency Exchange Rate", {})
            price = float(rate_data.get("5. Exchange Rate") or 0)
            if price:
                # Alpha Vantage doesn't give % change for forex — calculate from bid/ask
                return {"price": f"{price:,.2f}", "change": "0.00%"}
            return None

        # Crypto (e.g., BTC-USD → symbol=BTC, market=USD)
        if "-USD" in sym:
            crypto = sym.split("-")[0]
            url = (
                "https://www.alphavantage.co/query"
                f"?function=CURRENCY_EXCHANGE_RATE"
                f"&from_currency={crypto}&to_currency=USD"
                f"&apikey={api_key}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.load(resp)
            rate_data = data.get("Realtime Currency Exchange Rate", {})
            price = float(rate_data.get("5. Exchange Rate") or 0)
            if price:
                return {"price": f"{price:,.2f}", "change": "0.00%"}
            return None

        # Stocks/ETFs/Indices — use GLOBAL_QUOTE
        # Strip ^ prefix (Alpha Vantage doesn't use it)
        av_sym = sym.replace("^", "").replace("=F", "")
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={urllib.parse.quote(av_sym)}"
            f"&apikey={api_key}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        quote = data.get("Global Quote", {})
        price = float(quote.get("05. price") or 0)
        perc = float((quote.get("10. change percent") or "0").replace("%", ""))
        if price:
            return {"price": f"{price:,.2f}", "change": f"{perc:+.2f}%"}
    except Exception as exc:
        _log("WARN", f"  Alpha Vantage ({sym}) failed: {exc}")
    return None


def _fetch_market_data() -> dict:
    """
    Fetch all market data in parallel.

    India:       NSE historical EOD → NSE intraday → Yahoo Finance → Finnhub
    Everything else: Yahoo Finance → Finnhub (parallel)

    Result keys (in display order):
      India:       "Nifty 50", "Sensex"
      US:          "S&P 500", "NASDAQ", "Dow Jones"
      Asia/Europe: "Nikkei 225", "FTSE 100", "DAX"
      Commodities: "Gold", "Silver", "Brent Crude"
      Crypto:      "Bitcoin"
    """
    result: dict = {}
    _NA = {"price": "[N/A]", "change": "[N/A]"}

    # ── India: official exchange APIs first ──────────────────────────────────
    # Fetch both from official sources first
    nifty = _fetch_nse_nifty()
    sensex = _fetch_bse_sensex()

    # If BOTH official sources work, use them (consistent same-session data)
    if nifty and sensex:
        result["Nifty 50"] = nifty
        result["Sensex"] = sensex
        _log("DATA", f"  Nifty 50 (NSE): {nifty['price']} ({nifty['change']})")
        _log("DATA", f"  Sensex (BSE): {sensex['price']} ({sensex['change']})")
    else:
        # If either fails, use same source for BOTH to ensure consistent % change
        _log("INFO", "  NSE/BSE incomplete — using Finnhub/AV/Yahoo for both ...")
        q_nifty = (_fetch_finnhub_quote("^NSEI") or
                   _fetch_alphavantage_quote("^NSEI") or
                   _fetch_yahoo_quote("^NSEI"))
        q_sensex = (_fetch_finnhub_quote("^BSESN") or
                    _fetch_alphavantage_quote("^BSESN") or
                    _fetch_yahoo_quote("^BSESN"))
        result["Nifty 50"] = q_nifty or _NA
        result["Sensex"] = q_sensex or _NA
        if q_nifty:
            _log("DATA", f"  Nifty 50 (fallback): {q_nifty['price']} ({q_nifty['change']})")
        if q_sensex:
            _log("DATA", f"  Sensex (fallback): {q_sensex['price']} ({q_sensex['change']})")

    # ── Global indices + commodities + crypto (parallel) ────────────────────
    _GLOBAL_SYMBOLS: list[tuple[str, str]] = [
        ("S&P 500",    "^GSPC"),
        ("NASDAQ",     "^IXIC"),
        ("Dow Jones",  "^DJI"),
        ("Nikkei 225", "^N225"),
        ("FTSE 100",   "^FTSE"),
        ("DAX",        "^GDAXI"),
        ("Gold",       "GC=F"),
        ("Silver",     "SI=F"),
        ("Brent Crude", "BZ=F"),
        ("Bitcoin",    "BTC-USD"),
        ("USD/INR",    "USDINR=X"),
    ]

    def _fetch_one(label: str, sym: str) -> tuple[str, Optional[dict]]:
        # Priority: Finnhub → Alpha Vantage → Yahoo
        q = _fetch_finnhub_quote(sym)
        if q:
            _log("DATA", f"  {label} (Finnhub): {q['price']} ({q['change']})")
            return label, q
        q = _fetch_alphavantage_quote(sym)
        if q:
            _log("DATA", f"  {label} (AlphaVantage): {q['price']} ({q['change']})")
            return label, q
        q = _fetch_yahoo_quote(sym)
        if q:
            _log("DATA", f"  {label} (Yahoo): {q['price']} ({q['change']})")
        else:
            _log("WARN", f"  {label} ({sym}): all sources failed")
        return label, q

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, lbl, sym): lbl for lbl, sym in _GLOBAL_SYMBOLS}
        for fut in as_completed(futures):
            label, q = fut.result()
            result[label] = q or _NA

    # Preserve display order
    ordered: dict = {}
    for key in [
        "Nifty 50", "Sensex", "USD/INR",
        "S&P 500", "NASDAQ", "Dow Jones",
        "Nikkei 225", "FTSE 100", "DAX",
        "Gold", "Silver", "Brent Crude",
        "Bitcoin",
    ]:
        ordered[key] = result.get(key, _NA)
    return ordered


def _fetch_hn_headlines(n: int = 8) -> list:
    """
    HackerNews official Firebase API.
    Running since 2013; stable indefinitely.
    Fetches extra IDs to account for filtered meta-posts (Ask/Show/Tell/Launch HN).
    Retries the topstories endpoint on transient failures.
    """
    try:
        ids = _retry(
            lambda: json.loads(
                urllib.request.urlopen(
                    "https://hacker-news.firebaseio.com/v0/topstories.json",
                    timeout=10,
                ).read()
            )[: n * 3],  # fetch 3× to account for filtered meta-posts
            attempts=3,
            base_delay=1.5,
        )
    except Exception as exc:
        _log("WARN", f"  HN topstories failed: {exc}")
        return []

    titles = []
    for story_id in ids:
        if len(titles) >= n:
            break
        try:
            with urllib.request.urlopen(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=5,
            ) as resp:
                item = json.load(resp)
            title = (item.get("title") or "").strip()
            # Skip meta-posts (Ask HN, Show HN, Tell HN, Launch HN) —
            # these are community discussions, not real news/tech items.
            if any(title.lower().startswith(p) for p in _HN_META_PREFIXES):
                continue
            if item.get("type") == "story" and title:
                titles.append(title)
        except Exception:
            pass
        time.sleep(0.05)  # gentle rate limiting

    _log("DATA", f"  HN: {len(titles)} headlines fetched")
    return titles


# ──────────────────────────────────────────────────────────────────────────────
# Search fallbacks — per-section chain: Exa → RSS → DDG Lite → Mojeek
# Exa is a paid API (EXA_API_KEY); RSS/DDG/Mojeek need no key.
# _fetch_free_search() encapsulates the full chain.
# ──────────────────────────────────────────────────────────────────────────────

def _is_junk_title(title: str) -> bool:
    """
    Return True if a search result title is a generic news-site homepage title
    rather than an actual article headline.
    Examples of junk: "Latest News Today, Top Headlines & Live Updates — News24"
    """
    t = title.lower()
    return any(frag in t for frag in _JUNK_TITLE_FRAGMENTS)


def _strip_html_brief(raw: str) -> str:
    """Strip HTML tags and entities from a short snippet string."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_lib.unescape(text)
    return " ".join(text.split())[:150]


def _fetch_rss_headlines(url: str, n: int = 4) -> list:
    """
    Fetch article headlines from an RSS/Atom feed.
    Uses feedparser — handles RSS 2.0, Atom, encoding issues, malformed XML.
    Returns list of "**Title** — brief snippet." strings.
    Falls back to [] on any error.
    """
    try:
        d = feedparser.parse(url, agent="Mozilla/5.0 (digest-bot/1.0)")
        if d.get("bozo") and not d.get("entries"):
            _log("WARN", f"  RSS {url}: unparseable — {d.get('bozo_exception')}")
            return []
        items: list[str] = []
        for entry in d.entries:
            if len(items) >= n:
                break
            title = (entry.get("title") or "").strip()
            if not title or _is_junk_title(title):
                continue
            desc_raw = (
                entry.get("summary")
                or (entry.get("content") or [{}])[0].get("value")
                or ""
            ).strip()
            brief = _strip_html_brief(desc_raw).split(". ")[0] if desc_raw else ""
            items.append(f"**{title}** — {brief}." if brief else f"**{title}**")
        if items:
            _log("DATA", f"  RSS: {len(items)} results — {url.split('/')[2]}")
        return items
    except Exception as exc:
        _log("WARN", f"  RSS {url}: {exc}")
        return []


def _fetch_rss_section(section: str, n: int = 4) -> list:
    """
    Try each RSS feed URL for the given section in order.
    Returns the first non-empty result list, or [] if all fail.
    """
    for url in _RSS_SECTION_FEEDS.get(section, []):
        results = _fetch_rss_headlines(url, n)
        if results:
            return results
    return []


def _fetch_ddg_headlines(query: str, n: int = 4) -> list:
    """
    Search via DuckDuckGo Lite (lite.duckduckgo.com/lite/).
    Simple POST, returns plain HTML — no JS, no cookies needed on clean IPs.
    Requires beautifulsoup4 (in requirements-digest.txt).
    Falls back gracefully to [] on any error.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        _log("WARN", "  DDG: beautifulsoup4 not installed")
        return []
    data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
    req = urllib.request.Request(
        "https://lite.duckduckgo.com/lite/",
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read()
        soup = BeautifulSoup(html, "html.parser")
        items = []
        for link in soup.find_all("a", class_="result-link"):
            if len(items) >= n:
                break
            title = link.get_text(strip=True)
            if not title or _is_junk_title(title):
                continue
            snippet = ""
            row = link.find_parent("tr")
            if row:
                next_row = row.find_next_sibling("tr")
                if next_row:
                    cells = next_row.find_all("td")
                    cell = cells[1] if len(cells) >= 2 else (cells[0] if cells else None)
                    if cell:
                        text = cell.get_text(strip=True)
                        if text and not text.startswith(("http", "www.")):
                            snippet = text[:150]
            items.append(f"**{title}** — {snippet}." if snippet else f"**{title}**")
        _log("DATA", f"  DDG: {len(items)} results (after junk filter)")
        return items
    except Exception as exc:
        _log("WARN", f"  DDG search failed: {exc}")
        return []


def _fetch_mojeek_headlines(query: str, n: int = 4) -> list:
    """
    Search via Mojeek (mojeek.com) — independent index, no CAPTCHA, no API key.
    Requires beautifulsoup4.
    Falls back gracefully to [] on any error.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        _log("WARN", "  Mojeek: beautifulsoup4 not installed")
        return []
    req = urllib.request.Request(
        "https://www.mojeek.com/search?" + urllib.parse.urlencode({"q": query, "num": n + 2}),
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read()
        soup = BeautifulSoup(html, "html.parser")
        items = []
        for item in soup.select("ul.results-standard li"):
            if len(items) >= n:
                break
            title_el = item.select_one("a.ob")
            if not title_el:
                continue
            span = title_el.select_one("span")
            if span:
                span.decompose()
            title = title_el.get_text(strip=True)
            if _is_junk_title(title):
                continue
            snippet_el = item.select_one("p.s")
            snippet = snippet_el.get_text(strip=True)[:150] if snippet_el else ""
            if title:
                items.append(f"**{title}** — {snippet}." if snippet else f"**{title}**")
        _log("DATA", f"  Mojeek: {len(items)} results (after junk filter)")
        return items
    except Exception as exc:
        _log("WARN", f"  Mojeek search failed: {exc}")
        return []


def _fetch_exa_headlines(query: str, n: int = 4) -> list:
    """
    Search via Exa deep neural search (api.exa.ai).
    Requires EXA_API_KEY and exa-py package.
    Uses type="deep", category="news", highlights for brief snippets.
    Falls back gracefully to [] on any error or missing key.
    """
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return []
    try:
        from exa_py import Exa
        client = Exa(api_key)
        # yesterday as start date to filter to today's news only
        yesterday = (_now - timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # search_and_contents() fetches text/highlights alongside results in one call.
        # text=True and highlights=True use SDK defaults (500 chars, 2 sentences).
        resp = client.search_and_contents(
            query,
            category="news",
            num_results=n + 2,
            type="deep",
            text=True,
            highlights=True,
            start_published_date=yesterday,
        )
        items = []
        for r in (resp.results or []):
            if len(items) >= n:
                break
            title = (r.title or "").strip()
            if not title:
                continue
            # prefer first highlight snippet over raw text (more relevant)
            brief = ""
            highlights = getattr(r, "highlights", None) or []
            text = getattr(r, "text", None) or ""
            if highlights:
                brief = str(highlights[0]).strip()[:150]
            elif text:
                brief = text.split(". ")[0].strip()[:150]
            items.append(f"**{title}** — {brief}." if brief else f"**{title}**")
        _log("DATA", f"  Exa: {len(items)} results")
        return items
    except Exception as exc:
        _log("WARN", f"  Exa search failed: {exc}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Third-party news APIs — four independent paid sources for max stability.
# Chain position: NewsAPI → GNews → Currents → Finnhub News → Exa → RSS → DDG → Mojeek
# All require API keys set as GitHub Secrets.
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_headline(title: str, desc: str = "") -> str:
    """
    Format a title + description into a digest headline string.
    Strips " - Source Name" suffixes that NewsAPI and others append to titles
    (e.g. "Tesla cuts prices — Reuters" → "Tesla cuts prices").
    """
    # Strip trailing " - Source" / " | Source" / " – Source" suffix.
    # Only strip if the part after the separator is ≤ 50 chars (source name, not content).
    for sep in (" - ", " | ", " – ", " — "):
        idx = title.rfind(sep)
        if 0 < idx <= len(title) - len(sep) and len(title) - idx - len(sep) <= 50:
            title = title[:idx].strip()
            break
    title = title.strip()
    if not title:
        return ""
    brief = _strip_html_brief(desc).split(". ")[0][:150] if desc else ""
    return f"**{title}** — {brief}." if brief else f"**{title}**"


# Section → NewsAPI /v2/top-headlines query parameters.
# Ref: https://newsapi.org/docs/endpoints/top-headlines
# country and category may be combined; cannot mix with sources.
_NEWSAPI_PARAMS: dict[str, dict] = {
    "global": {"language": "en",                           "pageSize": "8"},
    "india":  {"country": "in",                            "pageSize": "8"},
    "tech":   {"category": "technology", "language": "en", "pageSize": "8"},
}


def _fetch_newsapi(section: str = "global", n: int = 4) -> list:
    """
    Fetch headlines from NewsAPI.org /v2/top-headlines.
    Returns list of formatted headline strings, or [] on error/missing key.
    Ref: https://newsapi.org/docs/endpoints/top-headlines
    Response: {"status": "ok", "articles": [{"title", "description", "source", ...}]}
    """
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        return []
    params = dict(_NEWSAPI_PARAMS.get(section, _NEWSAPI_PARAMS["global"]))
    params["apiKey"] = api_key
    url = "https://newsapi.org/v2/top-headlines?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        if data.get("status") != "ok":
            _log("WARN", f"  NewsAPI [{section}]: status={data.get('status')} "
                         f"msg={data.get('message', '')[:80]}")
            return []
        items: list[str] = []
        for art in (data.get("articles") or []):
            if len(items) >= n:
                break
            title = (art.get("title") or "").strip()
            desc  = (art.get("description") or "").strip()
            if not title or _is_junk_title(title):
                continue
            h = _fmt_headline(title, desc)
            if h:
                items.append(h)
        _log("DATA", f"  NewsAPI [{section}]: {len(items)} results")
        return items
    except Exception as exc:
        _log("WARN", f"  NewsAPI [{section}] failed: {exc}")
        return []


# Section → GNews /api/v4/top-headlines query parameters.
# Ref: https://gnews.io/docs/v4#top-headlines
# category values: general, world, nation, business, technology,
#                  entertainment, sports, science, health
# country: ISO 3166-1 alpha-2 code; apikey: lowercase query param name
_GNEWS_PARAMS: dict[str, dict] = {
    "global": {"category": "world",                                   "lang": "en", "max": "8"},
    "india":  {"category": "nation",  "country": "in",               "lang": "en", "max": "8"},
    "tech":   {"category": "technology",                              "lang": "en", "max": "8"},
}


def _fetch_gnews(section: str = "global", n: int = 4) -> list:
    """
    Fetch headlines from GNews.io /api/v4/top-headlines.
    Returns list of formatted headline strings, or [] on error/missing key.
    Ref: https://gnews.io/docs/v4#top-headlines
    Response: {"articles": [{"title", "description", "source": {"name"}, ...}]}
    """
    api_key = os.environ.get("GNEWS_API_KEY", "")
    if not api_key:
        return []
    params = dict(_GNEWS_PARAMS.get(section, _GNEWS_PARAMS["global"]))
    params["apikey"] = api_key  # GNews uses lowercase 'apikey'
    url = "https://gnews.io/api/v4/top-headlines?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        items: list[str] = []
        for art in (data.get("articles") or []):
            if len(items) >= n:
                break
            title = (art.get("title") or "").strip()
            desc  = (art.get("description") or "").strip()
            if not title or _is_junk_title(title):
                continue
            h = _fmt_headline(title, desc)
            if h:
                items.append(h)
        _log("DATA", f"  GNews [{section}]: {len(items)} results")
        return items
    except Exception as exc:
        _log("WARN", f"  GNews [{section}] failed: {exc}")
        return []


# Section → Currents API /v1/latest-news query parameters.
# Ref: https://currentsapi.services/en/docs/
# category values: general, technology, national, world, finance,
#                  politics, science, sports, health, entertainment
# language: BCP 47 language tag (e.g. "en"); apiKey: mixed-case query param
_CURRENTS_PARAMS: dict[str, dict] = {
    "global": {"category": "world",    "language": "en", "page_size": "8"},
    "india":  {"category": "national", "language": "en", "page_size": "8"},
    "tech":   {"category": "technology","language": "en", "page_size": "8"},
}


def _fetch_currents(section: str = "global", n: int = 4) -> list:
    """
    Fetch headlines from Currents API /v1/latest-news.
    Returns list of formatted headline strings, or [] on error/missing key.
    Ref: https://currentsapi.services/en/docs/
    Response: {"status": "ok", "news": [{"title", "description", "published", ...}]}
    """
    api_key = os.environ.get("CURRENTS_API_KEY", "")
    if not api_key:
        return []
    params = dict(_CURRENTS_PARAMS.get(section, _CURRENTS_PARAMS["global"]))
    params["apiKey"] = api_key  # Currents uses mixed-case 'apiKey'
    url = "https://api.currentsapi.services/v1/latest-news?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        if data.get("status") != "ok":
            _log("WARN", f"  Currents [{section}]: status={data.get('status')}")
            return []
        items: list[str] = []
        for art in (data.get("news") or []):
            if len(items) >= n:
                break
            title = (art.get("title") or "").strip()
            desc  = (art.get("description") or "").strip()
            if not title or _is_junk_title(title):
                continue
            h = _fmt_headline(title, desc)
            if h:
                items.append(h)
        _log("DATA", f"  Currents [{section}]: {len(items)} results")
        return items
    except Exception as exc:
        _log("WARN", f"  Currents [{section}] failed: {exc}")
        return []


# Finnhub market news categories.
# Ref: https://finnhub.io/docs/api/market-news
# Available categories: general, forex, crypto, merger
# No dedicated technology category on Finnhub free tier.
_FINNHUB_NEWS_CAT: dict[str, str] = {
    "global": "general",
    "india":  "general",
    "tech":   "general",
}


def _fetch_finnhub_news(section: str = "global", n: int = 4) -> list:
    """
    Fetch market news from Finnhub /api/v1/news.
    Returns list of formatted headline strings, or [] on error/missing key.
    Ref: https://finnhub.io/docs/api/market-news
    Response: array of {headline, summary, source, url, datetime, category}
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return []
    category = _FINNHUB_NEWS_CAT.get(section, "general")
    url = f"https://finnhub.io/api/v1/news?category={category}&token={api_key}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            articles = json.load(resp)
        items: list[str] = []
        for art in (articles or []):
            if len(items) >= n:
                break
            title = (art.get("headline") or "").strip()
            desc  = (art.get("summary") or "").strip()
            if not title or _is_junk_title(title):
                continue
            h = _fmt_headline(title, desc)
            if h:
                items.append(h)
        _log("DATA", f"  Finnhub news [{section}]: {len(items)} results")
        return items
    except Exception as exc:
        _log("WARN", f"  Finnhub news [{section}] failed: {exc}")
        return []


# Mediastack API — section to category/country mapping
# Ref: https://mediastack.com/documentation
# Categories: general, business, entertainment, health, science, sports, technology
_MEDIASTACK_PARAMS: dict[str, dict] = {
    "global": {"categories": "general,business", "languages": "en", "limit": "8"},
    "india":  {"countries": "in", "languages": "en", "limit": "8"},
    "tech":   {"categories": "technology", "languages": "en", "limit": "8"},
}


def _fetch_mediastack(section: str = "global", n: int = 4) -> list:
    """
    Fetch news from Mediastack API.
    Free tier: 100 requests/month, 30-min delay, HTTP only.
    Ref: https://mediastack.com/documentation
    Response: {data: [{title, description, url, source, category, published_at}]}
    """
    api_key = os.environ.get("MEDIASTACK_API_KEY", "")
    if not api_key:
        return []
    params = _MEDIASTACK_PARAMS.get(section, _MEDIASTACK_PARAMS["global"])
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    # Free plan: HTTP only (no HTTPS)
    url = f"http://api.mediastack.com/v1/news?access_key={api_key}&{qs}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        items: list[str] = []
        for art in (data.get("data") or []):
            if len(items) >= n:
                break
            title = (art.get("title") or "").strip()
            desc = (art.get("description") or "").strip()
            if not title or _is_junk_title(title):
                continue
            h = _fmt_headline(title, desc)
            if h:
                items.append(h)
        _log("DATA", f"  Mediastack [{section}]: {len(items)} results")
        return items
    except Exception as exc:
        _log("WARN", f"  Mediastack [{section}] failed: {exc}")
        return []


# NYTimes Top Stories API — section mapping
# Ref: https://developer.nytimes.com/docs/top-stories-product/1/overview
# Sections: world, us, technology, business, science, health, sports, arts
_NYTIMES_SECTION: dict[str, str] = {
    "global": "world",
    "india":  "world",
    "tech":   "technology",
}


def _fetch_nytimes(section: str = "global", n: int = 4) -> list:
    """
    Fetch news from NYTimes. Tries Top Stories first, falls back to Most Popular.
    Free tier: 500 requests/day.

    Top Stories: https://api.nytimes.com/svc/topstories/v2/{section}.json
    Most Popular: https://api.nytimes.com/svc/mostpopular/v2/viewed/1.json
    Response: {results: [{title, abstract/url}]}
    """
    api_key = os.environ.get("NYTIMES_API_KEY", "")
    if not api_key:
        return []

    # Try Top Stories first (section-specific)
    nyt_section = _NYTIMES_SECTION.get(section, "world")
    url = f"https://api.nytimes.com/svc/topstories/v2/{nyt_section}.json?api-key={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        items: list[str] = []
        for art in (data.get("results") or []):
            if len(items) >= n:
                break
            title = (art.get("title") or "").strip()
            desc = (art.get("abstract") or "").strip()
            if not title or _is_junk_title(title):
                continue
            h = _fmt_headline(title, desc)
            if h:
                items.append(h)
        if items:
            _log("DATA", f"  NYTimes TopStories [{section}]: {len(items)} results")
            return items
    except Exception as exc:
        _log("WARN", f"  NYTimes TopStories [{section}] failed: {exc}")

    # Fallback: Most Popular (viewed in last day)
    url = f"https://api.nytimes.com/svc/mostpopular/v2/viewed/1.json?api-key={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        items = []
        for art in (data.get("results") or []):
            if len(items) >= n:
                break
            title = (art.get("title") or "").strip()
            desc = (art.get("abstract") or "").strip()
            if not title or _is_junk_title(title):
                continue
            h = _fmt_headline(title, desc)
            if h:
                items.append(h)
        _log("DATA", f"  NYTimes MostPopular: {len(items)} results")
        return items
    except Exception as exc:
        _log("WARN", f"  NYTimes MostPopular failed: {exc}")
        return []


def _fetch_free_search(query: str, n: int = 4, section: str = "global") -> list:
    """
    Per-section search fallback chain (no LLM).  Returns first non-empty result.

    Tier 1 — paid APIs (best freshness, structured data):
      NewsAPI → GNews → NYTimes → Currents → Mediastack → Finnhub News
    Tier 2 — Exa deep neural search (paid, requires EXA_API_KEY):
      Exa
    Tier 3 — free, no key (RSS feeds from reputable sources):
      RSS feeds
    Tier 4 — free scraping (quality lower, junk-filtered):
      DDG Lite → Mojeek

    `section` selects the right RSS feed list and API category params.
    """
    # Tier 1: third-party news APIs — all keyed, most reliable when available
    results = _fetch_newsapi(section, n)
    if results:
        return results
    results = _fetch_gnews(section, n)
    if results:
        return results
    results = _fetch_nytimes(section, n)
    if results:
        return results
    results = _fetch_currents(section, n)
    if results:
        return results
    results = _fetch_mediastack(section, n)
    if results:
        return results
    results = _fetch_finnhub_news(section, n)
    if results:
        return results
    # Tier 2: Exa — requires EXA_API_KEY
    results = _fetch_exa_headlines(query, n)
    if results:
        return results
    # Tier 3: RSS — free, no key, real article titles from reputable sources
    results = _fetch_rss_section(section, n)
    if results:
        return results
    _log("INFO", f"  RSS empty for [{section}] — trying DDG ...")
    # Tier 4: DDG Lite → Mojeek — free scraping, junk titles filtered
    results = _fetch_ddg_headlines(query, n)
    if results:
        return results
    _log("INFO", "  DDG empty — trying Mojeek ...")
    return _fetch_mojeek_headlines(query, n)


def _fetch_tavily_section(label: str, query: str,
                          topic: str = "news", n: int = 4) -> tuple:
    """
    Fetch news via Tavily SDK.
    Returns (answer: str, bullets: list[str]).
      answer  — Tavily's AI-synthesized paragraph (use as section intro/commentary)
      bullets — list of "**Title** — brief." strings for individual headlines
    Falls back to ("", []) on any error or missing key.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "", []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        resp = client.search(
            query=query,
            topic=topic,
            search_depth="advanced",
            include_answer="advanced",
            max_results=n + 2,      # fetch extras in case some have no title
            time_range="day",
        )
        answer  = (resp.get("answer") or "").strip()
        bullets = []
        for r in (resp.get("results") or []):
            if len(bullets) >= n:
                break
            title   = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()
            brief   = content.split(". ")[0][:150] if content else ""
            if title:
                bullets.append(f"**{title}** — {brief}." if brief else f"**{title}**")
        _log("DATA", f"  Tavily [{label}]: {len(bullets)} bullets"
                     f"{', answer ok' if answer else ''}")
        return answer, bullets
    except Exception as exc:
        _log("WARN", f"  Tavily [{label}] failed: {exc}")
        return "", []


def _build_direct(
    market: dict, hn: list,
    mkt_commentary: str,
    glob_news: list, india_news: list, tech_news: list,
) -> str:
    """
    Level 1.5: assemble a complete digest from pre-fetched data. No LLM, no new API calls.
    Returns empty string if no news data is available at all.
    """
    if not (glob_news or india_news or tech_news):
        _log("SKIP", "data-direct — no news data available, skipping Level 1.5")
        return ""

    def _mrow(key: str) -> str:
        v = market.get(key, {"price": "[N/A]", "change": "[N/A]"})
        return f"| {key} | {v['price']} | {v['change']} |"

    nifty  = market.get("Nifty 50", {"price": "[N/A]", "change": "[N/A]"})

    def _section(items: list, placeholder: str) -> str:
        return ("\n".join(f"- {b}" for b in items[:3])
                if items else placeholder)

    global_sec = _section(glob_news,  "- _No global news available today._")
    india_sec  = _section(india_news, "- _No India news available today._")
    tech_mixed = tech_news[:2] + [f"**{h}**" for h in hn[:2]]
    tech_sec   = _section(tech_mixed, "- _No tech news available today._")

    parts = []
    if nifty["price"] != "[N/A]":
        parts.append(f"Nifty {nifty['price']} ({nifty['change']})")
    if glob_news:
        parts.append(glob_news[0].replace("**", "").split(" — ")[0][:80])
    summary = "; ".join(parts) + "." if parts else "Daily markets and news digest."

    return f"""\
---
title: "Daily Digest — {DATE_HUMAN}"
date: {DATE_FRONT}
summary: "{summary}"
---

## Markets

**India**

| Index | Price | Change |
|-------|-------|--------|
{_mrow("Nifty 50")}
{_mrow("Sensex")}
{_mrow("USD/INR")}

**Global**

| Index | Price | Change |
|-------|-------|--------|
{_mrow("S&P 500")}
{_mrow("NASDAQ")}
{_mrow("Dow Jones")}
{_mrow("Nikkei 225")}
{_mrow("FTSE 100")}
{_mrow("DAX")}

**Commodities & Crypto**

| Asset | Price | Change |
|-------|-------|--------|
{_mrow("Gold")}
{_mrow("Silver")}
{_mrow("Brent Crude")}
{_mrow("Bitcoin")}

---

## Global News

{global_sec}

---

## India

{india_sec}

---

## AI & Tech

{tech_sec}"""


# ──────────────────────────────────────────────────────────────────────────────
# Level 1 — search-capable AI with pre-fetched context
# ──────────────────────────────────────────────────────────────────────────────

def _make_level1(prompt: str) -> list:
    """
    Level 1: BEST models with web search enabled.
    Ranked by quality and search capability.
    All have access to pre-fetched data AND can search for more.
    """
    def _gemini() -> str:
        """Gemini with Google Search grounding — best for real-time news."""
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options={"timeout": 120})
        resp = client.models.generate_content(
            model=CFG["GEMINI_MODEL"],
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
        return resp.text or ""

    def _openai() -> str:
        """OpenAI search-preview model — built-in web search."""
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120.0)
        resp = client.chat.completions.create(
            model=CFG["OPENAI_SEARCH_MODEL"],
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def _openrouter_search() -> str:
        """Perplexity via OpenRouter — native web search."""
        return _openai_compatible_call(
            "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
            "OPENROUTER_SEARCH_MODEL", prompt, timeout=120.0,
        )

    def _deepseek() -> str:
        """DeepSeek v4 — strong reasoning, uses pre-fetched data well."""
        return _openai_compatible_call(
            "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
            "DEEPSEEK_MODEL", prompt, timeout=120.0,
            extra_body={"thinking": {"type": "disabled"}},
        )

    def _xai() -> str:
        """xAI Grok — has real-time X/Twitter data access."""
        return _openai_compatible_call(
            "XAI_API_KEY", "XAI_BASE_URL",
            "XAI_MODEL", prompt, timeout=120.0,
        )

    def _claude() -> str:
        """Claude with web search tool."""
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=120.0)
        resp = client.messages.create(
            model=CFG["CLAUDE_MODEL"],
            max_tokens=2048,
            tools=[{"type": CFG["CLAUDE_SEARCH_TOOL"]}],
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    def _zai_search() -> str:
        """Z.AI GLM with web_search enabled."""
        api_key = os.environ.get("ZAI_API_KEY", "")
        if not api_key:
            return ""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=CFG["ZAI_BASE_URL"], timeout=120.0)
            resp = client.chat.completions.create(
                model=CFG["ZAI_MODEL"],
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search", "web_search": {"enable": True}}],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    # Ranked: Gemini (Google Search) > OpenAI (search-preview) > Perplexity >
    # Z.AI GLM (web_search) > DeepSeek > Grok (X data) > Claude
    return [
        ("gemini",     "GEMINI_API_KEY",     _gemini),
        ("openai",     "OPENAI_API_KEY",     _openai),
        ("openrouter", "OPENROUTER_API_KEY", _openrouter_search),
        ("zai",        "ZAI_API_KEY",        _zai_search),
        ("deepseek",   "DEEPSEEK_API_KEY",   _deepseek),
        ("xai",        "XAI_API_KEY",        _xai),
        ("claude",     "ANTHROPIC_API_KEY",  _claude),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Level 1.5 — direct assembly from pre-fetched data, no LLM
# ──────────────────────────────────────────────────────────────────────────────

def _make_level1_5(
    market: dict, hn: list,
    mkt_commentary: str,
    glob_news: list, india_news: list, tech_news: list,
) -> list:
    """
    Build the digest directly from pre-fetched data — no LLM, no extra API calls.
    key_env=None means always attempt (data availability checked inside).
    """
    def _direct() -> str:
        return _build_direct(market, hn, mkt_commentary, glob_news, india_news, tech_news)

    return [("data-direct", None, _direct)]


# ──────────────────────────────────────────────────────────────────────────────
# GitHub Models helper — stdlib urllib, no packages, GITHUB_TOKEN always set
# ──────────────────────────────────────────────────────────────────────────────

def _github_models_call(prompt: str) -> str:
    """
    Call GitHub Models via the OpenAI-compatible endpoint.
    Uses GITHUB_TOKEN (auto-injected in Actions, needs models: read permission).
    Pure stdlib — no extra packages required.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return ""
    # The REST API uses the bare model ID (e.g. "gpt-4o-mini"), not the
    # publisher-prefixed catalog ID ("openai/gpt-4o-mini") used by actions/ai-inference.
    model_id = CFG["GITHUB_MODEL"].split("/")[-1]
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }).encode()
    req = urllib.request.Request(
        f"{CFG['GITHUB_MODELS_BASE_URL']}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"] or ""


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 — standard AI with pre-fetched rich context (no extra search)
# ──────────────────────────────────────────────────────────────────────────────

def _make_level2(prompt: str) -> list:
    """Standard models — no web search, but prompt contains all pre-fetched data."""

    def _claude_data() -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=90.0)
        resp = client.messages.create(
            model=CFG["CLAUDE_MODEL"],
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    def _openai_data() -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=90.0)
        resp = client.chat.completions.create(
            model=CFG["OPENAI_MODEL"],
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def _gemini_data() -> str:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options={"timeout": 90})
        resp = client.models.generate_content(
            model=CFG["GEMINI_MODEL"],
            contents=prompt,
        )
        return resp.text or ""

    def _openrouter_data() -> str:
        return _openai_compatible_call(
            "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
            "OPENROUTER_FREE_MODEL", prompt,
        )

    def _github_models_data() -> str:
        return _github_models_call(prompt)

    def _deepseek_data() -> str:
        return _openai_compatible_call(
            "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
            "DEEPSEEK_MODEL", prompt,
            extra_body={"thinking": {"type": "disabled"}},
        )

    def _mistral_data() -> str:
        return _openai_compatible_call(
            "MISTRAL_API_KEY", "MISTRAL_BASE_URL",
            "MISTRAL_MODEL", prompt,
        )

    def _groq_data() -> str:
        return _openai_compatible_call(
            "GROQ_API_KEY", "GROQ_BASE_URL",
            "GROQ_MODEL", prompt, timeout=60.0,
        )

    def _xai_data() -> str:
        return _openai_compatible_call(
            "XAI_API_KEY", "XAI_BASE_URL",
            "XAI_MODEL", prompt,
        )

    def _fireworks_data() -> str:
        return _openai_compatible_call(
            "FIREWORKS_API_KEY", "FIREWORKS_BASE_URL",
            "FIREWORKS_MODEL", prompt,
        )

    def _moonshot_data() -> str:
        return _openai_compatible_call(
            "MOONSHOT_AI_API_KEY", "MOONSHOT_BASE_URL",
            "MOONSHOT_MODEL", prompt,
            extra_body={"thinking": {"type": "disabled"}},
        )

    def _minimax_data() -> str:
        return _openai_compatible_call(
            "MINIMAX_API_KEY", "MINIMAX_BASE_URL",
            "MINIMAX_MODEL", prompt,
        )

    def _zai_data() -> str:
        """Z.AI (GLM) — supports web_search tool via tools parameter."""
        api_key = os.environ.get("ZAI_API_KEY", "")
        if not api_key:
            return ""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=CFG["ZAI_BASE_URL"], timeout=90.0)
            resp = client.chat.completions.create(
                model=CFG["ZAI_MODEL"],
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search", "web_search": {"enable": True}}],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Retry without web_search if tool not supported
            return _openai_compatible_call(
                "ZAI_API_KEY", "ZAI_BASE_URL",
                "ZAI_MODEL", prompt,
            )

    # Ranked by output quality (best first):
    # Tier 1: Best reasoning/instruction-following
    # Tier 2: Fast, reliable
    # Tier 3: Free/fallback options
    return [
        ("gemini+data",     "GEMINI_API_KEY",      _gemini_data),
        ("openai+data",     "OPENAI_API_KEY",      _openai_data),
        ("deepseek+data",   "DEEPSEEK_API_KEY",    _deepseek_data),
        ("zai+data",        "ZAI_API_KEY",         _zai_data),
        ("claude+data",     "ANTHROPIC_API_KEY",   _claude_data),
        ("groq+data",       "GROQ_API_KEY",        _groq_data),
        ("xai+data",        "XAI_API_KEY",         _xai_data),
        ("mistral+data",    "MISTRAL_API_KEY",     _mistral_data),
        ("openrouter+data", "OPENROUTER_API_KEY",  _openrouter_data),
        ("fireworks+data",  "FIREWORKS_API_KEY",   _fireworks_data),
        ("moonshot+data",   "MOONSHOT_AI_API_KEY", _moonshot_data),
        ("minimax+data",    "MINIMAX_API_KEY",     _minimax_data),
        # Always available in GitHub Actions — final cloud fallback
        ("github-models",   "GITHUB_TOKEN",        _github_models_data),
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Level 3 — data-only, no LLM
# ──────────────────────────────────────────────────────────────────────────────

def _data_only(market: dict, hn: list,
               tavily_global: Optional[list] = None,
               tavily_india:  Optional[list] = None,
               tavily_tech:   Optional[list] = None) -> str:
    """
    Build a digest from fetched data. No LLM.
    When Tavily results are provided, all sections are filled with real news.
    When Tavily is absent, Global News and India sections show [verify] markers.
    """
    def _mrow(key: str) -> str:
        v = market.get(key, {"price": "[N/A]", "change": "[N/A]"})
        return f"| {key} | {v['price']} | {v['change']} |"

    nifty = market.get("Nifty 50", {"price": "[N/A]", "change": "[N/A]"})

    def _section_bullets(tavily: Optional[list], hn_items: list,
                         verify_msg: str) -> str:
        if tavily:
            return "\n".join(f"- {h}" for h in tavily[:3])
        if hn_items:
            return "\n".join(f"- **{h}**" for h in hn_items[:3])
        return verify_msg

    global_bullets = _section_bullets(
        tavily_global, [],
        "- **[verify]** — _Add today's global news._\n"
        "- **[verify]** — _Add today's global news._",
    )
    india_bullets = _section_bullets(
        tavily_india, [],
        "- **[verify]** — _Add today's India news._\n"
        "- **[verify]** — _Add today's India news._",
    )
    # Tech: prefer Tavily tech news, fall back to HN, then verify
    tech_hn = [f"**{h}**" for h in hn[:4]]
    tech_bullets = _section_bullets(
        (tavily_tech or [])[:2] + tech_hn[:2] if (tavily_tech or tech_hn) else None,
        tech_hn,
        "- **[verify]** — _Add tech/jobs news._",
    )

    parts = []
    if nifty["price"] != "[N/A]":
        parts.append(f"Nifty {nifty['price']} ({nifty['change']})")
    if tavily_global:
        first = tavily_global[0].replace("**", "").split(" — ")[0][:80]
        parts.append(first)
    elif hn:
        parts.append(hn[0][:80])
    summary = "; ".join(parts) + "." if parts else "[AUTO — verify content before publishing]"

    return f"""\
---
title: "Daily Digest — {DATE_HUMAN}"
date: {DATE_FRONT}
summary: "{summary}"
---

## Markets

**India**

| Index | Price | Change |
|-------|-------|--------|
{_mrow("Nifty 50")}
{_mrow("Sensex")}
{_mrow("USD/INR")}

**Global**

| Index | Price | Change |
|-------|-------|--------|
{_mrow("S&P 500")}
{_mrow("NASDAQ")}
{_mrow("Dow Jones")}
{_mrow("Nikkei 225")}
{_mrow("FTSE 100")}
{_mrow("DAX")}

**Commodities & Crypto**

| Asset | Price | Change |
|-------|-------|--------|
{_mrow("Gold")}
{_mrow("Silver")}
{_mrow("Brent Crude")}
{_mrow("Bitcoin")}

---

## Global News

{global_bullets}

---

## India

{india_bullets}

---

## AI & Tech

{tech_bullets}"""

# ──────────────────────────────────────────────────────────────────────────────
# Level 4 — blank template, zero dependencies, always succeeds
# ──────────────────────────────────────────────────────────────────────────────

def _template_only() -> str:
    """Pure Python stdlib. Never fails. Edit before publishing."""
    return f"""\
---
title: "Daily Digest — {DATE_HUMAN}"
date: {DATE_FRONT}
summary: "[DRAFT — fill in summary before publishing]"
---

## Markets

**India**

| Index | Price | Change |
|-------|-------|--------|
| Nifty 50 | [price] | [change]% |
| Sensex | [price] | [change]% |
| USD/INR | [price] | [change]% |

**Global**

| Index | Price | Change |
|-------|-------|--------|
| S&P 500 | [price] | [change]% |
| NASDAQ | [price] | [change]% |
| Dow Jones | [price] | [change]% |
| Nikkei 225 | [price] | [change]% |
| FTSE 100 | [price] | [change]% |
| DAX | [price] | [change]% |

**Commodities & Crypto**

| Asset | Price | Change |
|-------|-------|--------|
| Gold | [price] | [change]% |
| Silver | [price] | [change]% |
| Brent Crude | [price] | [change]% |
| Bitcoin | [price] | [change]% |

---

## Global News

- **[Headline]** — [detail].
- **[Headline]** — [detail].

---

## India

- **[Headline]** — [detail].
- **[Headline]** — [detail].

---

## AI & Tech

- **[Headline]** — [detail].
- **[Headline]** — [detail]."""

# ──────────────────────────────────────────────────────────────────────────────
# Provider runner
# ──────────────────────────────────────────────────────────────────────────────

def _run(providers: list) -> Optional[tuple]:
    """
    Try providers in order. Skip those with no API key.
    key_env=None means always attempt (used for data-direct which needs no key).
    Return (text, name) on first success, None if all fail.
    """
    for name, key_env, fn in providers:
        if key_env and not os.environ.get(key_env):
            _log("SKIP", f"{name} — {key_env} not set")
            continue
        try:
            _log("TRY", f"{name} ...")
            text = fn()
            if _validate(text):
                _log("OK", f"{name} ✓")
                return text, name
            snippet = (_normalize(text) or "")[:100].replace("\n", "↵")
            _log("FAIL", f"{name} — invalid output: {snippet!r}")
        except Exception as exc:
            _log("FAIL", f"{name} — {type(exc).__name__}: {exc}")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Parallel pre-fetch
# ──────────────────────────────────────────────────────────────────────────────

def _parallel_prefetch() -> dict:
    """
    Fetch all data sources concurrently using a thread pool.
    Returns a dict with keys: market, hn, mkt, glob, india, tech.
    Each value is the raw return of the corresponding fetch function.
    """
    tasks: dict[str, Callable] = {
        "market": _fetch_market_data,
        "hn":     _fetch_hn_headlines,
        "mkt":    lambda: _fetch_tavily_section(
            "finance",
            f"India Nifty Sensex stock market today {DATE_HUMAN}",
            topic="finance",
        ),
        "glob":   lambda: _fetch_tavily_section(
            "global", f"major world news today {DATE_HUMAN}",
        ),
        "india":  lambda: _fetch_tavily_section(
            "india", f"India economy politics business news {DATE_HUMAN}",
        ),
        "tech":   lambda: _fetch_tavily_section(
            "tech", f"AI technology startup jobs news {DATE_HUMAN}",
        ),
    }
    results: dict = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                _log("WARN", f"  parallel fetch [{key}] failed: {exc}")
                results[key] = None
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _log("START", f"Daily Digest generator — {DATE_HUMAN}")
    _log("START", f"Target: {OUTPUT_FILE}")

    # Idempotent — skip if already generated today
    if OUTPUT_FILE.exists():
        _log("SKIP", "File already exists — nothing to do.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result: Optional[str] = None
    source: str = "unknown"

    # ── Pre-fetch ALL data concurrently ────────────────────────────────────
    # All six fetches run in parallel — market, HN, and four Tavily sections.
    # Sequential total was ~15-25s; parallel total is ~max(individual) ~8-12s.
    _log("INFO", "─── Pre-fetching data (parallel) ────────────────────────")
    prefetch = _parallel_prefetch()

    market         = prefetch.get("market") or {}
    hn             = prefetch.get("hn") or []
    mkt_commentary = (prefetch.get("mkt") or ("", []))[0]
    glob_news      = (prefetch.get("glob") or ("", []))[1]
    india_news     = (prefetch.get("india") or ("", []))[1]
    tech_news      = (prefetch.get("tech") or ("", []))[1]

    # Per-section fallback: Exa → DDG → Mojeek (run in parallel if multiple needed)
    sections_needing_fallback: dict[str, Callable] = {}
    if not glob_news:
        sections_needing_fallback["glob"]  = lambda: _fetch_free_search(
            f"world news today {DATE_HUMAN}", section="global"
        )
    if not india_news:
        sections_needing_fallback["india"] = lambda: _fetch_free_search(
            f"India news today {DATE_HUMAN}", section="india"
        )
    if not tech_news:
        sections_needing_fallback["tech"]  = lambda: _fetch_free_search(
            f"AI technology news today {DATE_HUMAN}", section="tech"
        )

    if sections_needing_fallback:
        _log("INFO", f"  Tavily empty for: {list(sections_needing_fallback)} — running fallback")
        with ThreadPoolExecutor(max_workers=len(sections_needing_fallback)) as executor:
            future_to_key = {executor.submit(fn): key
                             for key, fn in sections_needing_fallback.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    res = future.result() or []
                    if key == "glob":
                        glob_news = res
                    elif key == "india":
                        india_news = res
                    elif key == "tech":
                        tech_news = res
                except Exception as exc:
                    _log("WARN", f"  fallback fetch [{key}] failed: {exc}")

    # Deduplicate — remove cross-section duplicates (same story in global + india, etc.)
    glob_news, india_news, tech_news = _dedup_news(glob_news, india_news, tech_news)

    _log("INFO", f"  Pre-fetch done: market={bool(market)}, hn={len(hn)}, "
                 f"global={len(glob_news)}, india={len(india_news)}, tech={len(tech_news)}")

    # Build prompts for Level 1 (search hint on) and Level 2/Ollama (self-contained)
    search_prompt = _prompt_with_rich_data(
        market, hn, glob_news, india_news, tech_news, mkt_commentary,
        search_hint=True,
    )
    data_prompt = _prompt_with_rich_data(
        market, hn, glob_news, india_news, tech_news, mkt_commentary,
        search_hint=False,
    )

    # ── Level 1: search-capable AI + pre-fetched context ───────────────────
    _log("INFO", "─── Level 1: AI + search + pre-fetched context ──────────")
    outcome = _run(_make_level1(search_prompt))
    if outcome:
        result, source = outcome

    # ── Level 1.5: direct assembly — no LLM ────────────────────────────────
    if not result:
        _log("INFO", "─── Level 1.5: direct assembly (no LLM) ─────────────")
        outcome = _run(_make_level1_5(
            market, hn, mkt_commentary, glob_news, india_news, tech_news,
        ))
        if outcome:
            result, source = outcome

    # ── Level 2: standard AI + pre-fetched rich context ────────────────────
    if not result:
        _log("INFO", "─── Level 2: standard AI + pre-fetched context ───────")
        outcome = _run(_make_level2(data_prompt))
        if outcome:
            result, source = outcome

    # ── Level 2.5: Local Ollama model ──────────────────────────────────────
    # Only active when OLLAMA_MODEL env var is set (by the workflow after it
    # detects all cloud APIs failed and installs Ollama as a fallback).
    # Receives the same rich pre-fetched context — biggest benefit here since
    # local models cannot search the web themselves.
    if not result:
        _log("INFO", "─── Level 2.5: local Ollama model ────────────────────")
        ollama_model = os.environ.get("OLLAMA_MODEL", "")
        if not ollama_model:
            _log("SKIP", "ollama — OLLAMA_MODEL not set")
        else:
            try:
                _log("TRY", f"ollama ({ollama_model}) ...")
                body = json.dumps({
                    "model": ollama_model,
                    "prompt": data_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 4096},
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read())
                text = data.get("response", "")
                if _validate(text):
                    result, source = text, "ollama"
                    _log("OK", f"ollama ({ollama_model}) ✓")
                else:
                    snippet = (_normalize(text) or "")[:100].replace("\n", "↵")
                    _log("FAIL", f"ollama — invalid output: {snippet!r}")
            except Exception as exc:
                _log("FAIL", f"ollama — {type(exc).__name__}: {exc}")

    # ── Level 3: data-only template — reuses pre-fetched news (no new calls) ──
    if not result:
        _log("INFO", "─── Level 3: data-only template ──────────────────────")
        try:
            candidate = _data_only(market, hn, glob_news, india_news, tech_news)
            if _validate(candidate):
                result, source = candidate, "data-only"
                _log("OK", "data-only template ✓")
            else:
                # _validate rejects [verify] markers — Level 3 falls through to Level 4
                # The workflow Ollama check will trigger on the [DRAFT markers below.
                _log("WARN", "data-only template contains placeholder markers — falling to Level 4")
        except Exception as exc:
            _log("FAIL", f"data-only failed: {exc}")

    # ── Level 4: blank template — always succeeds ──────────────────────────
    if not result:
        _log("INFO", "─── Level 4: blank template ──────────────────────────")
        result = _template_only()
        source = "blank-template"
        _log("OK", "blank template created — edit before publishing")

    # ── Replace Markets section with REAL data (never trust AI for numbers) ──
    def _build_real_markets(mkt: dict) -> str:
        """Build the Markets markdown section from actual fetched data."""
        def _r(key: str) -> str:
            v = mkt.get(key, {"price": "[N/A]", "change": "[N/A]"})
            return f"| {key} | {v['price']} | {v['change']} |"

        return f"""## Markets

**India**

| Index | Price | Change |
|-------|-------|--------|
{_r("Nifty 50")}
{_r("Sensex")}
{_r("USD/INR")}

**Global**

| Index | Price | Change |
|-------|-------|--------|
{_r("S&P 500")}
{_r("NASDAQ")}
{_r("Dow Jones")}
{_r("Nikkei 225")}
{_r("FTSE 100")}
{_r("DAX")}

**Commodities & Crypto**

| Asset | Price | Change |
|-------|-------|--------|
{_r("Gold")}
{_r("Silver")}
{_r("Brent Crude")}
{_r("Bitcoin")}"""

    # Replace AI-generated Markets section with real data
    if market:
        real_markets = _build_real_markets(market)
        # Find and replace: everything from "## Markets" to the next "---" or "## "
        markets_pattern = re.compile(
            r"## Markets.*?(?=\n---|\n## (?!Markets)|$)",
            re.DOTALL,
        )
        if markets_pattern.search(result):
            result = markets_pattern.sub(real_markets, result, count=1)
            _log("INFO", "  Replaced AI Markets section with real data")
        else:
            # Markets section missing — prepend it after front matter
            if "\n---\n" in result:
                # Insert after the closing --- of front matter
                fm_end = result.index("\n---\n", result.index("---") + 3) + 5
                result = result[:fm_end] + "\n" + real_markets + "\n\n---\n" + result[fm_end:]
                _log("INFO", "  Injected real Markets section (was missing)")

    # ── Append "Further Reading" links from RSS feeds ───────────────────
    def _fetch_further_reading(n: int = 5) -> str:
        """Fetch top headlines from diverse RSS feeds for Further Reading section."""
        import random
        links: list[tuple[str, str]] = []  # (title, url)
        feeds = list(_FURTHER_READING_FEEDS)  # Copy to avoid mutating module constant
        random.shuffle(feeds)
        for feed_url in feeds:
            if len(links) >= n:
                break
            try:
                req = urllib.request.Request(
                    feed_url,
                    headers={"User-Agent": "Mozilla/5.0 (digest-bot/1.0)"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = resp.read()
                feed = feedparser.parse(raw)
                for entry in (feed.entries or [])[:2]:
                    title = (entry.get("title") or "").strip()
                    link = (entry.get("link") or "").strip()
                    if title and link and len(links) < n:
                        # Skip duplicates
                        if not any(t == title for t, _ in links):
                            links.append((title, link))
            except Exception:
                continue
        if not links:
            return ""
        md = "\n\n---\n\n## Further Reading\n\n"
        md += "\n".join(f"- [{t}]({u})" for t, u in links[:n])
        return md

    further = _fetch_further_reading()
    if further:
        result = result.rstrip() + further
        _log("INFO", f"  Appended Further Reading ({further.count('- [')} links)")

    # ── Convert markdown to HTML and write ────────────────────────────────
    author = _SOURCE_AUTHOR.get(source, "")
    final_md = _inject_author(_clean(result), author)

    # Extract front matter for manifest, then strip it from the markdown body
    title = f"Daily Digest — {DATE_HUMAN}"
    summary = ""
    body_md = final_md
    if final_md.startswith("---"):
        parts = final_md.split("---", 2)
        if len(parts) >= 3:
            fm_block = parts[1]
            body_md = parts[2].strip()
            for line in fm_block.strip().splitlines():
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("summary:"):
                    summary = line.split(":", 1)[1].strip().strip('"')

    # Convert markdown body to HTML
    html_body = md_lib.markdown(
        body_md,
        extensions=["tables", "nl2br"],
        output_format="html",
    )

    # Post-process: add color classes to change column values
    def _colorize_change(m: re.Match) -> str:
        val = m.group(1)
        if val.startswith("+"):
            return f'<td class="change-positive">{val}</td>'
        elif val.startswith("-"):
            return f'<td class="change-negative">{val}</td>'
        return f"<td>{val}</td>"

    html_body = re.sub(
        r"<td>([+-][\d.]+%)</td>",
        _colorize_change,
        html_body,
    )

    OUTPUT_FILE.write_text(html_body, encoding="utf-8")
    _log("DONE", f"Written via [{source}]{f' · author: {author}' if author else ''} → {OUTPUT_FILE}")

    # ── Update manifest.json ───────────────────────────────────────────────
    manifest: list = []
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = []

    # Remove existing entry for today (in case of re-run)
    manifest = [e for e in manifest if e.get("date") != DATE_ISO]
    # Add new entry
    manifest.append({
        "date": DATE_ISO,
        "title": title,
        "summary": summary,
        "source": source,
    })
    # Sort descending
    manifest.sort(key=lambda e: e["date"], reverse=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _log("DONE", f"Manifest updated — {len(manifest)} entries")


def test_all() -> None:
    """
    Test ALL providers and report pass/fail for each.
    Usage: python scripts/generate_digest.py --test
    Does NOT write any output file. Just validates connectivity.
    """
    _log("TEST", "=" * 60)
    _log("TEST", "TESTING ALL PROVIDERS — verifying API keys and connectivity")
    _log("TEST", "=" * 60)

    # ── Test market data sources ──────────────────────────────────────────
    _log("TEST", "\n─── Market Data Sources ───")
    for name, fn in [
        ("Finnhub", lambda: _fetch_finnhub_quote("AAPL")),
        ("Alpha Vantage", lambda: _fetch_alphavantage_quote("AAPL")),
        ("Yahoo Finance", lambda: _fetch_yahoo_quote("AAPL")),
        ("NSE (Nifty)", _fetch_nse_nifty),
        ("BSE (Sensex)", _fetch_bse_sensex),
    ]:
        try:
            result = fn()
            if result:
                _log("PASS", f"  {name}: {result}")
            else:
                _log("FAIL", f"  {name}: returned None (key missing or API down)")
        except Exception as e:
            _log("FAIL", f"  {name}: {type(e).__name__}: {e}")

    # ── Test news data sources ────────────────────────────────────────────
    _log("TEST", "\n─── News Data Sources ───")
    for name, fn in [
        ("Tavily", lambda: _fetch_tavily_section("global", f"world news {DATE_HUMAN}")),
        ("NewsAPI", lambda: _fetch_newsapi("global", 2)),
        ("GNews", lambda: _fetch_gnews("global", 2)),
        ("NYTimes", lambda: _fetch_nytimes("global", 2)),
        ("Currents", lambda: _fetch_currents("global", 2)),
        ("Mediastack", lambda: _fetch_mediastack("global", 2)),
        ("Finnhub News", lambda: _fetch_finnhub_news("global", 2)),
        ("Exa", lambda: _fetch_exa_headlines("world news today", 2)),
        ("RSS (BBC)", lambda: _fetch_rss_section("global", 2)),
    ]:
        try:
            result = fn()
            if result:
                _log("PASS", f"  {name}: {len(result)} items — {result[0][:80]}...")
            else:
                _log("FAIL", f"  {name}: returned empty (key missing or API down)")
        except Exception as e:
            _log("FAIL", f"  {name}: {type(e).__name__}: {e}")

    # ── Test AI providers ─────────────────────────────────────────────────
    _log("TEST", "\n─── AI Providers (Level 1 — search-capable) ───")
    test_prompt = (
        "Respond with exactly: TEST_OK followed by today's date. "
        "Nothing else. No explanation."
    )

    # Level 1 providers
    level1 = _make_level1(test_prompt)
    for name, key_env, fn in level1:
        key = os.environ.get(key_env or "", "")
        if not key and key_env:
            _log("SKIP", f"  {name}: {key_env} not set")
            continue
        try:
            result = fn()
            if result and len(result) > 3:
                _log("PASS", f"  {name}: {result[:100]}")
            else:
                _log("FAIL", f"  {name}: empty or too short response")
        except Exception as e:
            _log("FAIL", f"  {name}: {type(e).__name__}: {e}")

    _log("TEST", "\n─── AI Providers (Level 2 — standard) ───")
    level2 = _make_level2(test_prompt)
    for name, key_env, fn in level2:
        key = os.environ.get(key_env or "", "")
        if not key and key_env:
            _log("SKIP", f"  {name}: {key_env} not set")
            continue
        try:
            result = fn()
            if result and len(result) > 3:
                _log("PASS", f"  {name}: {result[:100]}")
            else:
                _log("FAIL", f"  {name}: empty or too short response")
        except Exception as e:
            _log("FAIL", f"  {name}: {type(e).__name__}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    _log("TEST", "\n" + "=" * 60)
    _log("TEST", "TEST COMPLETE — check PASS/FAIL above for each provider")
    _log("TEST", "=" * 60)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_all()
    else:
        main()
