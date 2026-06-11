"""
setup_database.py
─────────────────
Creates tables and loads data into Neon PostgreSQL.

Tables:
  - stocks        : master ticker list
  - price_data    : daily OHLCV + log_return
  - sentiment_daily: aggregated daily sentiment per ticker

Input files:
  - data/processed/stock_prices_clean.csv
  - data/processed/sentiment_daily.csv

USAGE:
    python src/database/setup_database.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

# ── Connection ────────────────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL not found in .env file")

engine = create_engine(DB_URL)

# ── Tickers ───────────────────────────────────────────────────────────────────
TICKERS = {
    "AAPL"  : "Apple Inc.",
    "NVDA"  : "Nvidia Corporation",
    "AMZN"  : "Amazon.com Inc.",
    "GOOGL" : "Alphabet Inc.",
    "META"  : "Meta Platforms Inc.",
    "NFLX"  : "Netflix Inc.",
}

# ── File paths ────────────────────────────────────────────────────────────────
PRICES_PATH    = "data/processed/stock_prices_6tickers.csv"
SENTIMENT_PATH = "data/processed/sentiment_daily_6tickers.csv"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Create schema
# ══════════════════════════════════════════════════════════════════════════════
print("── CREATING SCHEMA ──────────────────────────────────────")

with engine.connect() as conn:

    # Drop existing tables
    conn.execute(text("""
        DROP TABLE IF EXISTS sentiment_daily CASCADE;
        DROP TABLE IF EXISTS price_data CASCADE;
        DROP TABLE IF EXISTS stocks CASCADE;
    """))

    # stocks — master ticker list
    conn.execute(text("""
        CREATE TABLE stocks (
            id           SERIAL PRIMARY KEY,
            ticker       VARCHAR(10) UNIQUE NOT NULL,
            company_name VARCHAR(100),
            created_at   TIMESTAMP DEFAULT NOW()
        );
    """))
    print("✓ Table created: stocks")

    # price_data — daily OHLCV per ticker
    conn.execute(text("""
        CREATE TABLE price_data (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(10) NOT NULL REFERENCES stocks(ticker),
            date        DATE NOT NULL,
            open_price  DECIMAL(10,4),
            high_price  DECIMAL(10,4),
            low_price   DECIMAL(10,4),
            close_price DECIMAL(10,4),
            volume      BIGINT,
            log_return  DECIMAL(10,6),
            UNIQUE (ticker, date)
        );
    """))
    print("✓ Table created: price_data")

    # sentiment_daily — aggregated sentiment per ticker per day
    conn.execute(text("""
        CREATE TABLE sentiment_daily (
            id            SERIAL PRIMARY KEY,
            ticker        VARCHAR(10) NOT NULL REFERENCES stocks(ticker),
            trading_date  DATE NOT NULL,
            avg_finbert   DECIMAL(8,4),
            std_finbert   DECIMAL(8,4),
            bullish_ratio DECIMAL(6,4),
            bearish_ratio DECIMAL(6,4),
            neutral_ratio DECIMAL(6,4),
            avg_prescore  DECIMAL(8,4),
            article_count INTEGER,
            UNIQUE (ticker, trading_date)
        );
    """))
    print("✓ Table created: sentiment_daily")

    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load data
# ══════════════════════════════════════════════════════════════════════════════
print("\n── LOADING DATA ─────────────────────────────────────────")

# ── Load prices ───────────────────────────────────────────────────────────────
prices = pd.read_csv(PRICES_PATH)
prices["date"] = pd.to_datetime(prices["date"]).dt.date

# ── Load sentiment ────────────────────────────────────────────────────────────
sentiment = pd.read_csv(SENTIMENT_PATH)
sentiment["trading_date"] = pd.to_datetime(
    sentiment["trading_date"]
).dt.date

# Strip timezone if present
if hasattr(sentiment["trading_date"].iloc[0], 'tzinfo'):
    sentiment["trading_date"] = pd.to_datetime(
        sentiment["trading_date"]
    ).dt.tz_localize(None).dt.date

print(f"  Prices    : {len(prices):,} rows")
print(f"  Sentiment : {len(sentiment):,} rows")

# ── Insert tickers ────────────────────────────────────────────────────────────
with engine.connect() as conn:
    for ticker, company in TICKERS.items():
        conn.execute(text("""
            INSERT INTO stocks (ticker, company_name)
            VALUES (:ticker, :company)
            ON CONFLICT (ticker) DO NOTHING;
        """), {"ticker": ticker, "company": company})
    conn.commit()

print(f"✓ Inserted {len(TICKERS)} tickers into stocks table")

# ── Insert price data ─────────────────────────────────────────────────────────
price_rows = prices[[
    "ticker", "date", "open_price", "high_price",
    "low_price", "close_price", "volume", "log_return"
]].copy()
price_rows = price_rows.where(pd.notna(price_rows), None)

price_rows.to_sql(
    "price_data", engine,
    if_exists="append", index=False,
    method="multi", chunksize=500
)
print(f"✓ Inserted {len(price_rows):,} rows into price_data")

# ── Insert sentiment data ─────────────────────────────────────────────────────
sentiment = sentiment.where(pd.notna(sentiment), None)

sentiment.to_sql(
    "sentiment_daily", engine,
    if_exists="append", index=False,
    method="multi", chunksize=500
)
print(f"✓ Inserted {len(sentiment):,} rows into sentiment_daily")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Validation
# ══════════════════════════════════════════════════════════════════════════════
print("\n── VALIDATION ───────────────────────────────────────────")

with engine.connect() as conn:

    # Row counts
    print("\n  Row counts:")
    for table in ["stocks", "price_data", "sentiment_daily"]:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count  = result.fetchone()[0]
        print(f"    {table:<20} {count:>6,} rows")

    # Date range per ticker — prices
    print("\n  Price data date range:")
    result = conn.execute(text("""
        SELECT ticker, MIN(date), MAX(date), COUNT(*)
        FROM price_data
        GROUP BY ticker
        ORDER BY ticker;
    """))
    for row in result:
        print(f"    {row[0]}: {row[1]} → {row[2]} ({row[3]:,} trading days)")

    # Date range per ticker — sentiment
    print("\n  Sentiment data date range:")
    result = conn.execute(text("""
        SELECT ticker, MIN(trading_date), MAX(trading_date), COUNT(*)
        FROM sentiment_daily
        GROUP BY ticker
        ORDER BY ticker;
    """))
    for row in result:
        print(f"    {row[0]}: {row[1]} → {row[2]} ({row[3]:,} days)")

    # Sample join
    print("\n  Sample join — last 3 AAPL days:")
    result = conn.execute(text("""
        SELECT
            p.ticker, p.date, p.close_price, p.log_return,
            s.avg_finbert, s.bullish_ratio, s.article_count
        FROM price_data p
        JOIN sentiment_daily s
            ON p.ticker = s.ticker
            AND p.date  = s.trading_date
        WHERE p.ticker = 'AAPL'
        ORDER BY p.date DESC
        LIMIT 3;
    """))
    rows = result.fetchall()
    print(f"    {'ticker':<8} {'date':<12} {'close':>8} {'log_ret':>9} "
          f"{'finbert':>9} {'bull%':>7} {'articles':>9}")
    print(f"    {'-'*68}")
    for row in rows:
        print(f"    {row[0]:<8} {str(row[1]):<12} {float(row[2]):>8.2f} "
              f"{float(row[3]) if row[3] else 0:>9.4f} "
              f"{float(row[4]):>9.4f} {float(row[5]):>7.4f} "
              f"{row[6]:>9}")

    # Gap check
    print("\n  Date alignment check (prices with no sentiment):")
    result = conn.execute(text("""
        SELECT p.ticker, COUNT(*) as gap_count
        FROM price_data p
        LEFT JOIN sentiment_daily s
            ON p.ticker = s.ticker
            AND p.date  = s.trading_date
        WHERE s.trading_date IS NULL
        GROUP BY p.ticker
        ORDER BY gap_count DESC;
    """))
    gaps = result.fetchall()
    if gaps:
        print(f"    Ticker   Gap days")
        for g in gaps:
            print(f"    {g[0]:<8} {g[1]:>4} days with no sentiment")
    else:
        print(f"    ✓ All price dates have matching sentiment rows")

print("\n── DONE ─────────────────────────────────────────────────")
print("  Database ready for feature engineering")
print("─" * 55)