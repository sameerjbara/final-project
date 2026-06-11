"""
clean_prices.py
───────────────
Cleans raw stock price data from Yahoo Finance (yfinance).

Input  : data/raw/stock_prices_raw.csv
Output : data/processed/stock_prices_clean.csv

Steps:
  1. Rename columns to lowercase
  2. Parse date
  3. Sort by ticker + date
  4. Drop missing close prices
  5. Add log_return per ticker
  6. Round price columns
  7. Validate
  8. Save

USAGE:
    python src/processing/clean_prices.py
"""

import pandas as pd
import numpy as np
import os


# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/raw/stock_prices_raw.csv"
OUTPUT_PATH = "data/processed/stock_prices_clean.csv"

EXPECTED_TICKERS = [
    "AAPL", "NVDA", "MSFT", "AMZN",
    "GOOGL", "META", "JPM", "NFLX",
]


def clean_prices(
    input_path  : str = INPUT_PATH,
    output_path : str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Cleans raw price data and saves to output_path.
    Returns cleaned DataFrame.
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print("── STARTING PRICE CLEANING ──────────────────────────────")
    print(f"  Input  : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Tickers: {sorted(df['ticker'].unique().tolist())}")

    # ── 1. Rename columns ─────────────────────────────────────────────────────
    rename_map = {
        "Date"   : "date",
        "Open"   : "open_price",
        "High"   : "high_price",
        "Low"    : "low_price",
        "Close"  : "close_price",
        "Volume" : "volume",
        "open"   : "open_price",
        "high"   : "high_price",
        "low"    : "low_price",
        "close"  : "close_price",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                             if k in df.columns})
    print(f"\n  Columns after rename: {list(df.columns)}")

    # ── 2. Parse date ─────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  Date range: {min(df['date'])} to {max(df['date'])}")

    # ── 3. Sort ───────────────────────────────────────────────────────────────
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── 4. Drop missing close prices ──────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["close_price"])
    print(f"  Dropped {before - len(df)} rows with missing close price")

    # ── 5. Add log_return per ticker ──────────────────────────────────────────
    df["log_return"] = (
        df.groupby("ticker")["close_price"]
          .transform(lambda x: np.log(x / x.shift(1)))
          .round(6)
    )
    print(f"  log_return computed")

    # ── 6. Round price columns ────────────────────────────────────────────────
    for col in ["open_price", "high_price", "low_price", "close_price"]:
        if col in df.columns:
            df[col] = df[col].round(4)

    # ── 7. Select final columns ───────────────────────────────────────────────
    FINAL_COLS = [
        "ticker", "date", "open_price", "high_price",
        "low_price", "close_price", "volume", "log_return"
    ]
    final_cols = [c for c in FINAL_COLS if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    # ── 8. Validation ─────────────────────────────────────────────────────────
    print(f"\n── VALIDATION ───────────────────────────────────────────")

    actual_tickers  = sorted(df["ticker"].unique().tolist())
    missing_tickers = [t for t in EXPECTED_TICKERS if t not in actual_tickers]
    extra_tickers   = [t for t in actual_tickers if t not in EXPECTED_TICKERS]

    if missing_tickers:
        print(f"  Missing tickers: {missing_tickers}")
    if extra_tickers:
        print(f"  Extra tickers  : {extra_tickers}")
    if not missing_tickers and not extra_tickers:
        print(f"  All {len(EXPECTED_TICKERS)} expected tickers present")

    print(f"\n  Per-ticker summary:")
    counts = []
    for ticker, grp in df.groupby("ticker"):
        count = len(grp)
        counts.append(count)
        print(f"    {ticker:<6} {count:>4} rows  "
              f"({grp['date'].min()} to {grp['date'].max()})")

    if len(set(counts)) == 1:
        print(f"\n  All tickers have same row count ({counts[0]})")
    else:
        print(f"\n  Row count mismatch: {counts}")

    print(f"\n  Null counts:")
    for col, n in df.isnull().sum().items():
        flag = "OK" if n == 0 or col == "log_return" else "PROBLEM"
        print(f"    {flag} {col:<15} {n} nulls")

    print(f"\n  log_return stats:")
    clean_returns = df["log_return"].dropna()
    print(f"    Mean : {clean_returns.mean():.6f}")
    print(f"    Std  : {clean_returns.std():.6f}")
    print(f"    Min  : {clean_returns.min():.6f}")
    print(f"    Max  : {clean_returns.max():.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n── SAVED ────────────────────────────────────────────────")
    print(f"  Output : {len(df):,} rows x {df.shape[1]} cols")
    print(f"  Path   : {output_path}")
    print("-" * 55)

    return df


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_clean = clean_prices()
    print("\nFirst 3 rows:")
    print(df_clean.head(3).to_string(index=False))