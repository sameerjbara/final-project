"""
guardian_collector.py
─────────────────────
Pulls news articles from The Guardian API for multiple tickers.
- Searches for articles mentioning each company by name/ticker
- Pulls month by month across a date range
- Saves progress after each month (safe to stop and resume)
- Rotates through multiple API keys when rate limit is hit
- 500 requests/day free tier, history back to 2009

USAGE:
    # Pull default tickers for default date range
    python src/collection/guardian_collector.py

    # Pull specific tickers only
    python src/collection/guardian_collector.py --tickers AAPL NVDA

    # Expand date range (skips already-pulled months)
    python src/collection/guardian_collector.py --start 2024-05

    # Merge all monthly files into one CSV
    python src/collection/guardian_collector.py --merge

    # Full run + merge in one command
    python src/collection/guardian_collector.py --merge --start 2024-05

    # Check coverage without pulling
    python src/collection/guardian_collector.py --coverage

    # Pull specific dates for one ticker
    python src/collection/guardian_collector.py --tickers AAPL --dates 2025-02-02 2025-02-09
"""

import requests
import pandas as pd
import time
import os
import argparse
import calendar
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://content.guardianapis.com/search"
DELAY      = 0.5      # seconds between requests (500/day = very generous)
PAGE_SIZE  = 50       # articles per page (max 50 for free tier)

# Default date range
START_DATE = "2024-05"
END_DATE   = "2026-05"

# Output paths
SAVE_DIR    = "data/raw/guardian_monthly"
OUTPUT_FILE = "data/raw/guardian_raw.csv"

# Tickers and their search terms
# Guardian doesn't use ticker symbols — we search by company name + keywords
TICKERS = [
    "AAPL",
    "NVDA",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "NFLX",
]

COMPANY_NAMES = {
    "AAPL"  : "Apple",
    "NVDA"  : "Nvidia",
    "MSFT"  : "Microsoft",
    "AMZN"  : "Amazon",
    "GOOGL" : "Alphabet",
    "META"  : "Meta",
    "JPM"   : "JPMorgan",
    "NFLX"  : "Netflix",
}

# Search queries per ticker
# Multiple keywords increase article coverage
SEARCH_QUERIES = {
    "AAPL"  : "Apple AND (stock OR earnings OR revenue OR iPhone)",
    "NVDA"  : "Nvidia AND (stock OR earnings OR GPU OR AI)",
    "MSFT"  : "Microsoft AND (stock OR earnings OR Azure OR cloud)",
    "AMZN"  : "Amazon AND (stock OR earnings OR AWS OR revenue)",
    "GOOGL" : "Google AND (stock OR earnings OR revenue OR advertising)",
    "META"  : "Meta AND (stock OR earnings OR revenue OR advertising)",
    "JPM"   : "JPMorgan AND (stock OR earnings OR revenue OR banking)",
    "NFLX"  : "Netflix AND (stock OR earnings OR subscribers OR revenue)",
}

# Guardian sections most relevant for financial news
FINANCIAL_SECTIONS = "business,technology,money"


# ── API Key rotation ──────────────────────────────────────────────────────────
# Add GUARDIAN_API_KEY_1, GUARDIAN_API_KEY_2 etc. to your .env file
def load_api_keys() -> list:
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GUARDIAN_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    if not keys:
        raise ValueError(
            "No Guardian API keys found. "
            "Add GUARDIAN_API_KEY_1 to your .env file. "
            "Get a free key at: open-platform.theguardian.com/access"
        )
    return keys

API_KEYS        = load_api_keys()
current_key_idx = 0

def get_current_key() -> str:
    return API_KEYS[current_key_idx]

def rotate_key() -> bool:
    """Rotate to next key. Returns True if succeeded, False if all exhausted."""
    global current_key_idx
    if current_key_idx < len(API_KEYS) - 1:
        current_key_idx += 1
        print(f"    ↻ Rotated to API key {current_key_idx + 1}/{len(API_KEYS)}")
        return True
    else:
        print(f"    ✖ All {len(API_KEYS)} API keys exhausted for today")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def get_month_range(year: int, month: int):
    """Returns (from_date, to_date) as YYYY-MM-DD strings."""
    first_day = 1
    last_day  = calendar.monthrange(year, month)[1]
    from_date = f"{year}-{month:02d}-{first_day:02d}"
    to_date   = f"{year}-{month:02d}-{last_day:02d}"
    return from_date, to_date


def get_months_in_range(start: str, end: str) -> list:
    """Returns list of (year, month) tuples between start and end inclusive."""
    start_year, start_month = map(int, start.split("-"))
    end_year,   end_month   = map(int, end.split("-"))
    months = []
    y, m   = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m  = 1
            y += 1
    return months


def save_path(ticker: str, year: int, month: int) -> str:
    return os.path.join(SAVE_DIR, f"{ticker}_{year}{month:02d}.csv")


def already_pulled(ticker: str, year: int, month: int) -> bool:
    return os.path.exists(save_path(ticker, year, month))


def coverage_report(tickers: list, start: str, end: str):
    months = get_months_in_range(start, end)
    total  = len(tickers) * len(months)
    pulled = sum(
        1 for y, m in months for t in tickers
        if already_pulled(t, y, m)
    )
    print(f"\n── COVERAGE REPORT ──────────────────────────────────────")
    print(f"  Range   : {start} → {end} ({len(months)} months)")
    print(f"  Tickers : {tickers}")
    print(f"  Pulled  : {pulled}/{total} months ({pulled/total*100:.0f}%)")
    print(f"  Missing : {total - pulled} months still to pull")
    print(f"\n  Per-ticker status:")
    for ticker in tickers:
        t_pulled  = sum(1 for y, m in months if already_pulled(ticker, y, m))
        t_missing = len(months) - t_pulled
        status    = "✓ complete" if t_missing == 0 else f"⚠ {t_missing} months missing"
        print(f"    {ticker:<6} {t_pulled:>2}/{len(months)} months  {status}")


# ══════════════════════════════════════════════════════════════════════════════
# CORE PULL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def fetch_page(
    query     : str,
    from_date : str,
    to_date   : str,
    page      : int = 1,
) -> dict | None:
    """
    Fetches one page of Guardian search results.
    Returns response dict or None on error.
    """
    while True:
        params = {
            "q"           : query,
            "from-date"   : from_date,
            "to-date"     : to_date,
            "page-size"   : PAGE_SIZE,
            "page"        : page,
            "show-fields" : "trailText,bodyText,headline",
            "order-by"    : "oldest",
            "api-key"     : get_current_key(),
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
        except Exception as e:
            print(f"\n    ✖ Request failed: {e}")
            return None

        # Rate limit
        if response.status_code == 429:
            print(f"\n    ⚠ Rate limit on key {current_key_idx + 1}")
            if rotate_key():
                time.sleep(5)
                continue
            else:
                return None

        # Auth error
        if response.status_code == 401:
            print(f"\n    ✖ Invalid key {current_key_idx + 1} — rotating")
            if rotate_key():
                continue
            else:
                return None

        if response.status_code != 200:
            print(f"\n    ✖ HTTP {response.status_code}")
            return None

        return response.json()


def pull_month(ticker: str, year: int, month: int) -> pd.DataFrame | None:
    """
    Pulls one month of Guardian articles for one ticker.
    Paginates through all available pages.
    Returns DataFrame on success, None on unrecoverable error.
    """
    from_date, to_date = get_month_range(year, month)
    query   = SEARCH_QUERIES.get(ticker, COMPANY_NAMES.get(ticker, ticker))
    company = COMPANY_NAMES.get(ticker, ticker)

    all_articles = []
    page         = 1
    total_pages  = 1   # updated after first response

    while page <= total_pages:
        data = fetch_page(query, from_date, to_date, page)

        if data is None:
            return None   # unrecoverable error

        response_data = data.get("response", {})
        status        = response_data.get("status", "")

        if status != "ok":
            print(f"\n    ✖ API error: {response_data.get('message', 'unknown')}")
            return None

        # Update total pages from first response
        if page == 1:
            total_pages = response_data.get("pages", 1)

        results = response_data.get("results", [])

        for article in results:
            fields = article.get("fields", {})
            all_articles.append({
                "ticker"      : ticker,
                "company"     : company,
                "source"      : "The Guardian",
                "title"       : fields.get("headline", article.get("webTitle", "")),
                "description" : fields.get("trailText", ""),
                "url"         : article.get("webUrl", ""),
                "publishedAt" : article.get("webPublicationDate", ""),
                "content"     : fields.get("bodyText", "")[:1000],  # first 1000 chars
                "data_source" : "Guardian",
                "date"        : f"{year}-{month:02d}",
                "section"     : article.get("sectionName", ""),
            })

        page += 1
        time.sleep(DELAY)

    return pd.DataFrame(all_articles) if all_articles else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# PULL SPECIFIC DATES
# ══════════════════════════════════════════════════════════════════════════════

def pull_specific_dates(ticker: str, dates: list) -> pd.DataFrame:
    """
    Pulls Guardian articles for a specific ticker on specific dates.

    Args:
        ticker : ticker symbol e.g. "AAPL"
        dates  : list of date strings "YYYY-MM-DD"

    Usage:
        df = pull_specific_dates("AAPL", ["2025-02-02", "2025-02-09"])
    """
    ensure_dirs()
    query      = SEARCH_QUERIES.get(ticker, COMPANY_NAMES.get(ticker, ticker))
    company    = COMPANY_NAMES.get(ticker, ticker)
    all_frames = []

    print(f"── PULLING SPECIFIC DATES ───────────────────────────────")
    print(f"  Ticker : {ticker}")
    print(f"  Query  : {query}")
    print(f"  Dates  : {dates}")
    print()

    for date_str in dates:
        print(f"  {date_str} pulling...", end=" ", flush=True)

        data = fetch_page(query, date_str, date_str, page=1)

        if data is None:
            print(f"✖ Failed")
            continue

        response_data = data.get("response", {})
        results       = response_data.get("results", [])

        if not results:
            print(f"0 articles")
            time.sleep(DELAY)
            continue

        articles = []
        for article in results:
            fields = article.get("fields", {})
            articles.append({
                "ticker"      : ticker,
                "company"     : company,
                "source"      : "The Guardian",
                "title"       : fields.get("headline", article.get("webTitle", "")),
                "description" : fields.get("trailText", ""),
                "url"         : article.get("webUrl", ""),
                "publishedAt" : article.get("webPublicationDate", ""),
                "content"     : fields.get("bodyText", "")[:1000],
                "data_source" : "Guardian",
                "date"        : date_str,
                "section"     : article.get("sectionName", ""),
            })

        df = pd.DataFrame(articles)
        all_frames.append(df)
        print(f"{len(df):>4} articles")
        time.sleep(DELAY)

    if not all_frames:
        print("  ✖ No articles collected")
        return pd.DataFrame()

    result      = pd.concat(all_frames, ignore_index=True)
    output_path = os.path.join("data/raw", f"{ticker}_guardian_specific_dates.csv")
    result.to_csv(output_path, index=False)

    print(f"\n── DONE ─────────────────────────────────────────────────")
    print(f"  Total articles : {len(result):,}")
    print(f"  Saved to       : {output_path}")
    print(f"\n  Per-date breakdown:")
    for date_str in dates:
        cnt = len(result[result["date"] == date_str])
        print(f"    {date_str}  →  {cnt} articles")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_collection(
    tickers : list = TICKERS,
    start   : str  = START_DATE,
    end     : str  = END_DATE,
) -> bool:
    """
    Pulls Guardian articles month by month for all tickers.
    Returns True if completed fully, False if stopped early.
    """
    ensure_dirs()
    months = get_months_in_range(start, end)

    print(f"── GUARDIAN COLLECTION ──────────────────────────────────")
    print(f"  API keys : {len(API_KEYS)} key(s) loaded from .env")
    print(f"  Tickers  : {tickers}")
    print(f"  Range    : {start} → {end} ({len(months)} months)")
    print(f"  Total    : {len(tickers) * len(months)} pulls needed")
    print(f"  Note     : Multiple pages pulled per month automatically")
    print(f"  Save dir : {SAVE_DIR}/")
    coverage_report(tickers, start, end)
    print()

    total_pulled   = 0
    total_skipped  = 0
    total_articles = 0

    for year, month in months:
        month_label = f"{year}-{month:02d}"
        print(f"  ── {month_label} ──────────────────────────────────────")

        for ticker in tickers:

            if already_pulled(ticker, year, month):
                print(f"    {ticker:<6} ✓ already saved — skipping")
                total_skipped += 1
                continue

            print(f"    {ticker:<6} pulling...", end=" ", flush=True)
            df = pull_month(ticker, year, month)

            if df is None:
                print(f"\n  ⚠ Stopping — all keys exhausted on {ticker} {month_label}")
                print(f"  Run the same command again to resume.")
                print(f"  Progress: {total_pulled} pulls, "
                      f"{total_articles:,} articles saved")
                return False

            if df.empty:
                print(f"0 articles (no news this month)")
                path = save_path(ticker, year, month)
                df.to_csv(path, index=False)
                total_pulled += 1
                continue

            path = save_path(ticker, year, month)
            df.to_csv(path, index=False)
            print(f"{len(df):>4} articles → {path}")
            total_pulled   += 1
            total_articles += len(df)

        print()

    print(f"── COLLECTION COMPLETE ──────────────────────────────────")
    print(f"  Pulled   : {total_pulled} months")
    print(f"  Skipped  : {total_skipped} months (already existed)")
    print(f"  Articles : {total_articles:,} total new articles")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MERGE ALL MONTHLY FILES
# ══════════════════════════════════════════════════════════════════════════════

def merge_monthly_files(
    tickers : list = TICKERS,
    start   : str  = START_DATE,
    end     : str  = END_DATE,
    output  : str  = OUTPUT_FILE,
) -> pd.DataFrame:
    """Merges all saved monthly CSV files into one raw DataFrame."""
    months  = get_months_in_range(start, end)
    frames  = []
    missing = []

    print(f"── MERGING MONTHLY FILES ────────────────────────────────")
    print(f"  Range   : {start} → {end}")
    print(f"  Tickers : {tickers}")

    for year, month in months:
        for ticker in tickers:
            path = save_path(ticker, year, month)
            if os.path.exists(path):
                if os.path.getsize(path) < 10:
                    continue   # skip empty files
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        frames.append(df)
                except Exception:
                    continue   # skip unreadable files
            else:
                missing.append(f"{ticker} {year}-{month:02d}")

    if missing:
        print(f"\n  ⚠ Missing {len(missing)} files (not yet pulled):")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    if not frames:
        print("  ✖ No files found to merge")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output, index=False)

    print(f"\n  ✓ Merged  : {len(frames)} files")
    print(f"  ✓ Rows    : {len(merged):,}")
    print(f"  ✓ Saved   : {output}")
    print(f"\n  Per-ticker article counts:")
    for ticker, cnt in merged["ticker"].value_counts().items():
        print(f"    {ticker:<6} {cnt:>6,} articles")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pull Guardian news data")
    parser.add_argument("--tickers", nargs="+", default=TICKERS,
                        help="Tickers to pull")
    parser.add_argument("--start", default=START_DATE,
                        help=f"Start month YYYY-MM (default: {START_DATE})")
    parser.add_argument("--end", default=END_DATE,
                        help=f"End month YYYY-MM (default: {END_DATE})")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all monthly files after collection")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip collection — just merge existing files")
    parser.add_argument("--coverage", action="store_true",
                        help="Show coverage report without pulling")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Pull specific dates for one ticker "
                             "e.g. --tickers AAPL --dates 2025-02-02 2025-02-09")
    args = parser.parse_args()

    if args.dates:
        if len(args.tickers) != 1:
            print("  ✖ --dates requires exactly one --tickers value")
            print("  Example: python src/collection/guardian_collector.py "
                  "--tickers AAPL --dates 2025-02-02 2025-02-09")
        else:
            pull_specific_dates(ticker=args.tickers[0], dates=args.dates)

    elif args.coverage:
        coverage_report(args.tickers, args.start, args.end)

    elif args.merge_only:
        merge_monthly_files(tickers=args.tickers, start=args.start, end=args.end)

    else:
        completed = run_collection(
            tickers=args.tickers, start=args.start, end=args.end
        )
        if completed and args.merge:
            print()
            merge_monthly_files(tickers=args.tickers, start=args.start, end=args.end)
        elif completed:
            print(f"\n  Tip: run with --merge to combine all files:")
            print(f"  python src/collection/guardian_collector.py --merge")