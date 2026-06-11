"""
filter_tickers.py
─────────────────
Removes JPM and MSFT from price and sentiment files.
Saves new filtered files for active use.
Original files kept untouched as backup.

Input  (backup):
  data/processed/stock_prices_clean.csv
  data/processed/sentiment_daily.csv

Output (active):
  data/processed/stock_prices_6tickers.csv
  data/processed/sentiment_daily_6tickers.csv

USAGE:
    python src/processing/filter_tickers.py
"""

import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────
EXCLUDE = ['JPM', 'MSFT']
KEEP    = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NFLX', 'NVDA']

PRICES_IN    = "data/processed/stock_prices_clean.csv"
SENTIMENT_IN = "data/processed/sentiment_daily.csv"

PRICES_OUT    = "data/processed/stock_prices_6tickers.csv"
SENTIMENT_OUT = "data/processed/sentiment_daily_6tickers.csv"

print("── FILTER TICKERS ───────────────────────────────────────")
print(f"  Removing  : {EXCLUDE}")
print(f"  Keeping   : {KEEP}")

# ── Prices ────────────────────────────────────────────────────────────────────
prices_orig = pd.read_csv(PRICES_IN)
prices_new  = prices_orig[~prices_orig['ticker'].isin(EXCLUDE)].copy()
prices_new.to_csv(PRICES_OUT, index=False)

print(f"\n  Prices:")
print(f"    Original : {len(prices_orig):,} rows  ({PRICES_IN})")
print(f"    Filtered : {len(prices_new):,} rows  ({PRICES_OUT})")
print(f"    Removed  : {len(prices_orig)-len(prices_new):,} rows")

# ── Sentiment ─────────────────────────────────────────────────────────────────
sent_orig = pd.read_csv(SENTIMENT_IN)
sent_new  = sent_orig[~sent_orig['ticker'].isin(EXCLUDE)].copy()
sent_new.to_csv(SENTIMENT_OUT, index=False)

print(f"\n  Sentiment:")
print(f"    Original : {len(sent_orig):,} rows  ({SENTIMENT_IN})")
print(f"    Filtered : {len(sent_new):,} rows  ({SENTIMENT_OUT})")
print(f"    Removed  : {len(sent_orig)-len(sent_new):,} rows")

# ── Verify ────────────────────────────────────────────────────────────────────
print(f"\n  Verification:")
print(f"    Prices tickers    : {sorted(prices_new['ticker'].unique().tolist())}")
print(f"    Sentiment tickers : {sorted(sent_new['ticker'].unique().tolist())}")

print(f"\n── SAVED ────────────────────────────────────────────────")
print(f"  Backup  (unchanged) : {PRICES_IN}")
print(f"  Backup  (unchanged) : {SENTIMENT_IN}")
print(f"  Active  (6 tickers) : {PRICES_OUT}")
print(f"  Active  (6 tickers) : {SENTIMENT_OUT}")
print("-" * 55)