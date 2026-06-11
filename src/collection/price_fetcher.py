"""
collect_prices.py
─────────────────
Pulls daily OHLCV price data from Yahoo Finance for multiple tickers.

USAGE:
    # Default — pull configured tickers for configured date range
    python collect_prices.py

    # Add a new ticker without re-pulling existing ones
    python collect_prices.py --tickers TSLA

    # Expand date range (only pulls missing dates)
    python collect_prices.py --start 2024-05-01

    # Pull completely fresh
    python collect_prices.py --fresh
"""

import yfinance as yf
import pandas as pd
import os
import argparse
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
# Add or remove tickers here
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

# Default date range
START_DATE = "2024-05-01"
END_DATE   = "2026-05-31"

# Output paths
RAW_DIR      = "data/raw"
OUTPUT_FILE  = "data/processed/stock_prices_raw.csv"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def raw_path(ticker: str) -> str:
    """Returns path for a ticker's individual raw CSV."""
    return os.path.join(RAW_DIR, f"{ticker}_prices.csv")


def already_pulled(ticker: str) -> bool:
    """Returns True if this ticker already has a saved raw file."""
    return os.path.exists(raw_path(ticker))


def pull_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Pulls daily OHLCV data for one ticker from Yahoo Finance.
    Returns cleaned DataFrame or None on failure.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"  ⚠ {ticker}: no data returned")
            return None

        # Fix MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.rename(columns={"Date": "Date"})
        df["ticker"] = ticker

        return df

    except Exception as e:
        print(f"  ✖ {ticker}: failed — {e}")
        return None


def expand_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    If ticker already exists, only pulls dates not already in the file.
    Useful for expanding date range without re-pulling everything.
    """
    path = raw_path(ticker)

    if not os.path.exists(path):
        # No existing file — pull fresh
        return pull_ticker(ticker, start, end)

    existing = pd.read_csv(path)
    existing["Date"] = pd.to_datetime(existing["Date"])

    existing_start = existing["Date"].min()
    existing_end   = existing["Date"].max()

    new_frames = []

    # Pull earlier dates if needed
    if pd.Timestamp(start) < existing_start:
        print(f"  ↑ {ticker}: pulling earlier dates "
              f"({start} → {existing_start.date()})")
        earlier = pull_ticker(ticker, start,
                              existing_start.strftime("%Y-%m-%d"))
        if earlier is not None:
            new_frames.append(earlier)

    # Pull later dates if needed
    if pd.Timestamp(end) > existing_end:
        print(f"  ↓ {ticker}: pulling later dates "
              f"({existing_end.date()} → {end})")
        later = pull_ticker(ticker, existing_end.strftime("%Y-%m-%d"), end)
        if later is not None:
            new_frames.append(later)

    if not new_frames:
        print(f"  ✓ {ticker}: already up to date — skipping")
        return existing

    # Combine existing + new, drop duplicates
    combined = pd.concat([existing] + new_frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = (combined
                .drop_duplicates(subset=["Date", "ticker"])
                .sort_values("Date")
                .reset_index(drop=True))

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_collection(
    tickers : list = TICKERS,
    start   : str  = START_DATE,
    end     : str  = END_DATE,
    fresh   : bool = False,
) -> pd.DataFrame:
    """
    Pulls price data for all tickers.

    Args:
        tickers : list of ticker symbols to pull
        start   : start date "YYYY-MM-DD"
        end     : end date "YYYY-MM-DD"
        fresh   : if True, re-pulls all tickers even if files exist

    Returns:
        Combined DataFrame with all tickers stacked vertically
    """

    ensure_dirs()

    print("── PRICE DATA COLLECTION ────────────────────────────────")
    print(f"  Tickers : {tickers}")
    print(f"  Range   : {start} → {end}")
    print(f"  Mode    : {'Fresh pull' if fresh else 'Incremental (skip existing)'}")
    print()

    all_frames = []
    pulled     = 0
    skipped    = 0
    failed     = 0

    for ticker in tickers:
        path = raw_path(ticker)

        if fresh or not already_pulled(ticker):
            # Fresh pull
            print(f"  Pulling {ticker}...", end=" ", flush=True)
            df = pull_ticker(ticker, start, end)

            if df is not None:
                df.to_csv(path, index=False)
                print(f"{len(df):,} rows → saved to {path}")
                all_frames.append(df)
                pulled += 1
            else:
                failed += 1

        else:
            # Incremental — only pull missing date ranges
            df = expand_ticker(ticker, start, end)

            if df is not None:
                df.to_csv(path, index=False)
                all_frames.append(df)
                skipped += 1
            else:
                failed += 1

    if not all_frames:
        print("  ✖ No data collected")
        return pd.DataFrame()

    # Combine all tickers
    combined = pd.concat(all_frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = (combined
                .sort_values(["ticker", "Date"])
                .reset_index(drop=True))

    # Save combined file
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\n── COLLECTION COMPLETE ──────────────────────────────────")
    print(f"  Pulled  : {pulled} tickers")
    print(f"  Skipped : {skipped} tickers (already up to date)")
    print(f"  Failed  : {failed} tickers")
    print(f"  Total   : {len(combined):,} rows")
    print(f"  Saved   : {OUTPUT_FILE}")

    print(f"\n  Per-ticker summary:")
    for ticker, grp in combined.groupby("ticker"):
        print(f"    {ticker:<6} {len(grp):>4} rows  "
              f"({grp['Date'].min().date()} → {grp['Date'].max().date()})")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# ADD TICKERS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def add_tickers(
    new_tickers : list,
    start       : str = START_DATE,
    end         : str = END_DATE,
) -> pd.DataFrame:
    """
    Adds new tickers to existing data without re-pulling current ones.

    Usage:
        add_tickers(["TSLA", "NFLX"])
    """
    print(f"── ADDING NEW TICKERS: {new_tickers} ────────────────────")

    # Load existing combined file
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        existing["Date"] = pd.to_datetime(existing["Date"])
        existing_tickers = existing["ticker"].unique().tolist()
        print(f"  Existing tickers: {existing_tickers}")
    else:
        existing = pd.DataFrame()
        print(f"  No existing file found — pulling fresh")

    # Only pull tickers not already in the file
    to_pull = [t for t in new_tickers
               if t not in (existing["ticker"].unique()
                            if not existing.empty else [])]

    if not to_pull:
        print(f"  All tickers already exist — nothing to pull")
        return existing

    print(f"  Pulling: {to_pull}")
    new_df = run_collection(tickers=to_pull, start=start, end=end)

    if new_df.empty:
        return existing

    # Combine and save
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = (combined
                .drop_duplicates(subset=["Date", "ticker"])
                .sort_values(["ticker", "Date"])
                .reset_index(drop=True))

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✓ Combined file updated: {len(combined):,} rows")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pull stock price data from Yahoo Finance"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Specific tickers to add (e.g. --tickers TSLA NFLX)"
    )
    parser.add_argument(
        "--start", default=START_DATE,
        help=f"Start date (default: {START_DATE})"
    )
    parser.add_argument(
        "--end", default=END_DATE,
        help=f"End date (default: {END_DATE})"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Re-pull all tickers even if files exist"
    )
    args = parser.parse_args()

    if args.tickers:
        # Adding specific new tickers
        df = add_tickers(
            new_tickers=args.tickers,
            start=args.start,
            end=args.end,
        )
    else:
        # Pull all configured tickers
        df = run_collection(
            tickers=TICKERS,
            start=args.start,
            end=args.end,
            fresh=args.fresh,
        )

    print(f"\n  Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")