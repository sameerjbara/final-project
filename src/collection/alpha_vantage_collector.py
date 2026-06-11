"""
alpha_vantage_collector.py
──────────────────────────
Pulls news sentiment data from Alpha Vantage for multiple tickers.
- Pulls month by month across a date range
- Saves progress after each month (safe to stop and resume)
- Handles rate limits gracefully
- Easy to expand date range or add tickers without re-pulling existing data

USAGE:
    # Pull default tickers for default date range
    python src/collection/alpha_vantage_collector.py

    # Pull specific tickers only
    python src/collection/alpha_vantage_collector.py --tickers NFLX AMD

    # Expand to 2 years (skips already-pulled months)
    python src/collection/alpha_vantage_collector.py --start 2024-06

    # Merge all monthly files into one CSV after collection
    python src/collection/alpha_vantage_collector.py --merge

    # Full run + merge in one command
    python src/collection/alpha_vantage_collector.py --merge --start 2024-06

    # Check what's already been pulled
    python src/collection/alpha_vantage_collector.py --coverage
"""

import requests
import pandas as pd
import time
import os
import argparse
import calendar
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.alphavantage.co/query"
DELAY    = 12        # seconds between requests (free tier: 5 req/min)
LIMIT    = 1000      # max articles per request

# ── API Key rotation ──────────────────────────────────────────────────────────
# Keys loaded from .env — add AV_API_KEY_1, AV_API_KEY_2, etc.
# Script rotates to next key automatically when rate limit is hit.
def load_api_keys() -> list:
    keys = []
    i = 1
    while True:
        key = os.getenv(f"AV_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    if not keys:
        raise ValueError(
            "No API keys found. Add AV_API_KEY_1, AV_API_KEY_2 etc. to your .env file"
        )
    return keys

API_KEYS        = load_api_keys()
current_key_idx = 0   # tracks which key we are currently using

def get_current_key() -> str:
    return API_KEYS[current_key_idx]

def rotate_key() -> bool:
    """
    Rotate to next API key.
    Returns True if rotation succeeded, False if all keys are exhausted.
    """
    global current_key_idx
    if current_key_idx < len(API_KEYS) - 1:
        current_key_idx += 1
        print(f"    ↻ Rotated to API key {current_key_idx + 1}/{len(API_KEYS)}")
        return True
    else:
        print(f"    ✖ All {len(API_KEYS)} API keys exhausted for today")
        return False

# Default date range
START_DATE = "2024-05"  
END_DATE   = "2026-05"   

# Output paths
SAVE_DIR    = "data/raw/av_monthly"
OUTPUT_FILE = "data/raw/alphavantage_raw.csv"

# Tickers — add or remove here
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def get_month_range(year: int, month: int):
    """
    Returns (from_date, to_date) strings for Alpha Vantage API.
    Format: YYYYMMDDTHHMM
    """
    first_day = 1
    last_day  = calendar.monthrange(year, month)[1]
    from_date = f"{year}{month:02d}{first_day:02d}T0000"
    to_date   = f"{year}{month:02d}{last_day:02d}T2359"
    return from_date, to_date


def get_months_in_range(start: str, end: str):
    """
    Returns list of (year, month) tuples between start and end inclusive.
    start / end format: "YYYY-MM"  e.g. "2025-06"
    """
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
    """Returns the file path for a ticker's monthly raw data."""
    return os.path.join(SAVE_DIR, f"{ticker}_{year}{month:02d}.csv")


def already_pulled(ticker: str, year: int, month: int) -> bool:
    """Returns True if this ticker+month has already been pulled and saved."""
    return os.path.exists(save_path(ticker, year, month))


def coverage_report(tickers: list, start: str, end: str):
    """Prints a summary of what's already been pulled vs what's missing."""
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

    # Show missing per ticker
    print(f"\n  Per-ticker status:")
    for ticker in tickers:
        t_pulled  = sum(1 for y, m in months if already_pulled(ticker, y, m))
        t_missing = len(months) - t_pulled
        status    = "✓ complete" if t_missing == 0 else f"⚠ {t_missing} months missing"
        print(f"    {ticker:<6} {t_pulled:>2}/{len(months)} months  {status}")


# ══════════════════════════════════════════════════════════════════════════════
# CORE PULL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def pull_month(ticker: str, year: int, month: int) -> pd.DataFrame | None:
    """
    Pulls one month of news for one ticker from Alpha Vantage.
    Returns DataFrame on success, None on rate limit or error.
    """
    from_date, to_date = get_month_range(year, month)

    # Try each available API key until one works
    while True:
        params = {
            "function"  : "NEWS_SENTIMENT",
            "tickers"   : ticker,
            "time_from" : from_date,
            "time_to"   : to_date,
            "sort"      : "LATEST",
            "limit"     : LIMIT,
            "apikey"    : get_current_key(),
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            data     = response.json()
        except Exception as e:
            print(f"    ✖ Request failed: {e}")
            return None

        # Rate limit hit — try rotating to next key
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            print(f"    ⚠ Rate limit on key {current_key_idx + 1}: {msg[:60]}")
            if rotate_key():
                time.sleep(2)   # brief pause before retrying with new key
                continue        # retry with new key
            else:
                return None     # all keys exhausted
        break   # successful response — exit the while loop

    # No feed in response
    if "feed" not in data:
        print(f"    ✖ No feed: {list(data.keys())}")
        return None

    articles = []
    company  = COMPANY_NAMES.get(ticker, ticker)

    for article in data["feed"]:
        articles.append({
            "ticker"                  : ticker,
            "company"                 : company,
            "source"                  : article.get("source"),
            "title"                   : article.get("title"),
            "description"             : article.get("summary"),
            "url"                     : article.get("url"),
            "publishedAt"             : article.get("time_published"),
            "content"                 : article.get("summary"),
            "overall_sentiment_score" : article.get("overall_sentiment_score"),
            "overall_sentiment_label" : article.get("overall_sentiment_label"),
            "data_source"             : "AlphaVantage",
            "date"                    : f"{year}-{month:02d}",
        })

    return pd.DataFrame(articles) if articles else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_collection(
    tickers : list = TICKERS,
    start   : str  = START_DATE,
    end     : str  = END_DATE,
) -> bool:
    """
    Pulls news month by month for all tickers between start and end.
    - Skips months already saved (safe to stop and resume)
    - Stops cleanly if rate limit is hit
    - Returns True if completed fully, False if stopped early

    Args:
        tickers : list of ticker symbols
        start   : start month "YYYY-MM"
        end     : end month "YYYY-MM"
    """
    ensure_dirs()
    months = get_months_in_range(start, end)

    print(f"── ALPHA VANTAGE COLLECTION ─────────────────────────────")
    print(f"  API keys : {len(API_KEYS)} key(s) loaded from .env")
    print(f"  Tickers  : {tickers}")
    print(f"  Range    : {start} → {end} ({len(months)} months)")
    print(f"  Total    : {len(tickers) * len(months)} pulls needed")
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

            # Skip if already pulled
            if already_pulled(ticker, year, month):
                print(f"    {ticker:<6} ✓ already saved — skipping")
                total_skipped += 1
                continue

            print(f"    {ticker:<6} pulling...", end=" ", flush=True)
            df = pull_month(ticker, year, month)

            # Rate limit — stop and tell user to resume later
            if df is None:
                print(f"\n  ⚠ Stopping — rate limit hit on {ticker} {month_label}")
                print(f"  Run the same command again tomorrow to resume.")
                print(f"  Already pulled months will be skipped automatically.")
                print(f"\n  Progress: {total_pulled} pulls, "
                      f"{total_articles:,} articles saved")
                return False

            # Empty response — no articles this month
            if df.empty:
                print(f"0 articles (no news this month)")
                path = save_path(ticker, year, month)
                df.to_csv(path, index=False)
                total_pulled += 1
                time.sleep(DELAY)
                continue

            # Save monthly file
            path = save_path(ticker, year, month)
            df.to_csv(path, index=False)
            print(f"{len(df):>4} articles → {path}")
            total_pulled   += 1
            total_articles += len(df)

            time.sleep(DELAY)

        print()

    print(f"── COLLECTION COMPLETE ──────────────────────────────────")
    print(f"  Pulled   : {total_pulled} months")
    print(f"  Skipped  : {total_skipped} months (already existed)")
    print(f"  Articles : {total_articles:,} total new articles")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MERGE ALL MONTHLY FILES INTO ONE CSV
# ══════════════════════════════════════════════════════════════════════════════

def merge_monthly_files(
    tickers : list = TICKERS,
    start   : str  = START_DATE,
    end     : str  = END_DATE,
    output  : str  = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Merges all saved monthly CSV files into one raw DataFrame.
    Call this after run_collection() finishes.
    Skips empty files silently.
    """
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
                df = pd.read_csv(path)
                if not df.empty:
                    frames.append(df)
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
# PULL SPECIFIC DATES FOR SPECIFIC TICKER
# ══════════════════════════════════════════════════════════════════════════════

def pull_specific_dates(
    ticker : str,
    dates  : list,
) -> pd.DataFrame:
    """
    Pulls news for a specific ticker on specific dates.
    Each date gets its own API call covering that full day.

    Args:
        ticker : ticker symbol e.g. "AAPL"
        dates  : list of date strings "YYYY-MM-DD"
                 e.g. ["2025-02-02", "2025-02-09"]

    Returns:
        DataFrame with all articles from those dates combined

    Usage:
        df = pull_specific_dates("AAPL", ["2025-02-02", "2025-02-09"])
    """
    ensure_dirs()

    print(f"── PULLING SPECIFIC DATES ───────────────────────────────")
    print(f"  Ticker : {ticker}")
    print(f"  Dates  : {dates}")
    print()

    all_frames = []

    for date_str in dates:
        # Parse date
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"  ✖ Invalid date format: {date_str} — use YYYY-MM-DD")
            continue

        # Build API date strings for that full day
        from_date = f"{dt.year}{dt.month:02d}{dt.day:02d}T0000"
        to_date   = f"{dt.year}{dt.month:02d}{dt.day:02d}T2359"

        params = {
            "function"  : "NEWS_SENTIMENT",
            "tickers"   : ticker,
            "time_from" : from_date,
            "time_to"   : to_date,
            "sort"      : "LATEST",
            "limit"     : LIMIT,
            "apikey"    : get_current_key(),
        }

        print(f"  {date_str} pulling...", end=" ", flush=True)

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            data     = response.json()
        except Exception as e:
            print(f"✖ Request failed: {e}")
            continue

        # Rate limit
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            print(f"✖ Rate limit: {msg[:60]}")
            print(f"\n  Stopped at {date_str} — resume later")
            break

        # No articles
        if "feed" not in data:
            print(f"0 articles")
            time.sleep(DELAY)
            continue

        company  = COMPANY_NAMES.get(ticker, ticker)
        articles = []

        for article in data["feed"]:
            articles.append({
                "ticker"                  : ticker,
                "company"                 : company,
                "source"                  : article.get("source"),
                "title"                   : article.get("title"),
                "description"             : article.get("summary"),
                "url"                     : article.get("url"),
                "publishedAt"             : article.get("time_published"),
                "content"                 : article.get("summary"),
                "overall_sentiment_score" : article.get("overall_sentiment_score"),
                "overall_sentiment_label" : article.get("overall_sentiment_label"),
                "data_source"             : "AlphaVantage",
                "date"                    : date_str,
            })

        df = pd.DataFrame(articles)
        all_frames.append(df)
        print(f"{len(df):>4} articles")

        time.sleep(DELAY)

    if not all_frames:
        print("  ✖ No articles collected")
        return pd.DataFrame()

    # Combine all dates
    result = pd.concat(all_frames, ignore_index=True)

    # Save to a specific-dates file
    output_name = f"{ticker}_specific_dates.csv"
    output_path = os.path.join("data/raw", output_name)
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
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pull Alpha Vantage news sentiment data"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=TICKERS,
        help=f"Tickers to pull (default: all configured)"
    )
    parser.add_argument(
        "--start", default=START_DATE,
        help=f"Start month YYYY-MM (default: {START_DATE})"
    )
    parser.add_argument(
        "--end", default=END_DATE,
        help=f"End month YYYY-MM (default: {END_DATE})"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all monthly files after collection"
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip collection — just merge existing files"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Show coverage report without pulling anything"
    )
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Pull specific dates for one ticker e.g. --tickers AAPL --dates 2025-02-02 2025-02-09"
    )
    args = parser.parse_args()

    if args.dates:
        if len(args.tickers) != 1:
            print("  -- --dates requires exactly one --tickers value")
            print("  Example: python src/collection/alpha_vantage_collector.py --tickers AAPL --dates 2025-02-02 2025-02-09")
        else:
            pull_specific_dates(ticker=args.tickers[0], dates=args.dates)

    elif args.coverage:
        coverage_report(args.tickers, args.start, args.end)

    elif args.merge_only:
        merge_monthly_files(
            tickers=args.tickers,
            start=args.start,
            end=args.end,
        )

    else:
        completed = run_collection(
            tickers=args.tickers,
            start=args.start,
            end=args.end,
        )

        if completed and args.merge:
            print()
            merge_monthly_files(
                tickers=args.tickers,
                start=args.start,
                end=args.end,
            )
        elif completed:
            print(f"\n  Tip: run with --merge to combine all files:")
            print(f"  python src/collection/alpha_vantage_collector.py --merge")