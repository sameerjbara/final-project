"""
feature_engineering_v2.py
──────────────────────────
Updated feature engineering based on EDA findings.

Changes from v1:
  REMOVED:
    - prescore_finbert_agreement  (weakest feature, broken on neutral-fill days)

  ADDED:
    - sentiment_lag4              (EDA: lag signal persists beyond 3 days)
    - sentiment_lag5              (EDA: lag signal persists beyond 3 days)
    - sentiment_acceleration      (EDA: trend of sentiment > level of sentiment)
    - article_count_norm          (replaces binary high_news_day with continuous)
    - article_count_change        (captures NVDA-style high-volume reversals)
    - volatility_sentiment        (interaction: volatility x sentiment)
    - prescore_finbert_diff       (replaces agreement flag with actual difference)

  KEPT:
    - All original lag/rolling features
    - All price features
    - high_news_day kept alongside article_count_norm for comparison
    - All calendar features

Output: data/processed/features_v2.csv

USAGE:
    python src/features/feature_engineering_v22.py
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

print("── FEATURE ENGINEERING v2 ───────────────────────────────")
print("  Changes based on EDA findings:")
print("  + sentiment_lag4, lag5       (lag signal persists)")
print("  + sentiment_acceleration     (trend > level)")
print("  + article_count_norm         (continuous vs binary)")
print("  + article_count_change       (high-volume reversal signal)")
print("  + volatility_sentiment       (interaction feature)")
print("  + prescore_finbert_diff      (replaces agreement flag)")
print("  - prescore_finbert_agreement (weakest, broken on neutral-fill)")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(4)


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).round(4)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load from PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 1 — Loading data from PostgreSQL...")

with engine.connect() as conn:
    prices = pd.read_sql("""
        SELECT ticker, date, open_price, high_price, low_price,
               close_price, volume, log_return
        FROM price_data
        ORDER BY ticker, date
    """, conn)

    sentiment = pd.read_sql("""
        SELECT ticker, trading_date, avg_finbert, std_finbert,
               bullish_ratio, bearish_ratio, neutral_ratio,
               avg_prescore, article_count
        FROM sentiment_daily
        ORDER BY ticker, trading_date
    """, conn)

prices["date"]            = pd.to_datetime(prices["date"])
sentiment["trading_date"] = pd.to_datetime(sentiment["trading_date"])

print(f"  Loaded {len(prices):,} price rows")
print(f"  Loaded {len(sentiment):,} sentiment rows")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Join prices + sentiment
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 2 — Joining prices + sentiment...")

df = prices.merge(
    sentiment,
    left_on  = ["ticker", "date"],
    right_on = ["ticker", "trading_date"],
    how      = "left"
).drop(columns=["trading_date"])

missing = df["avg_finbert"].isna().sum()
print(f"  Joined: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"  Rows with no sentiment: {missing:,} ({missing/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Fill missing sentiment
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 3 — Filling missing sentiment...")

NEUTRAL_FILL = {
    "avg_finbert"   : 0.0,
    "std_finbert"   : 0.0,
    "bullish_ratio" : 0.333,
    "bearish_ratio" : 0.333,
    "neutral_ratio" : 0.334,
    "avg_prescore"  : 0.0,
    "article_count" : 0,
}

for col, val in NEUTRAL_FILL.items():
    filled = df[col].isna().sum()
    df[col] = df[col].fillna(val)
    if filled > 0:
        print(f"    {col:<25} filled {filled:,} NaN -> {val}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Price features
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 4 — Computing price features...")

results = []
for ticker, group in df.groupby("ticker"):
    g = group.copy().sort_values("date").reset_index(drop=True)

    g["rsi_14"]          = compute_rsi(g["close_price"], period=14)
    g["price_momentum_5d"] = g["close_price"].pct_change(5).round(4)
    g["volatility_5d"]   = (g["log_return"]
                             .rolling(window=5, min_periods=2)
                             .std().round(6))
    g["macd"]            = compute_macd(g["close_price"])
    g["volume_change"]   = g["volume"].pct_change().round(4)
    g["return_lag1"]     = g["log_return"].shift(1).round(6)
    g["price_range"]     = (
        (g["high_price"] - g["low_price"]) / g["close_price"]
    ).round(4)

    results.append(g)

df = pd.concat(results, ignore_index=True)
print(f"  Price features computed")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Sentiment features (original + new EDA-driven)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 5 — Computing sentiment features...")

results = []
for ticker, group in df.groupby("ticker"):
    g = group.copy().sort_values("date").reset_index(drop=True)

    # ── Original lag features ─────────────────────────────────────────────────
    g["sentiment_lag1"] = g["avg_finbert"].shift(1).round(4)
    g["sentiment_lag2"] = g["avg_finbert"].shift(2).round(4)
    g["sentiment_lag3"] = g["avg_finbert"].shift(3).round(4)

    # ── NEW: Extended lag features (EDA: lag signal persists) ─────────────────
    g["sentiment_lag4"] = g["avg_finbert"].shift(4).round(4)
    g["sentiment_lag5"] = g["avg_finbert"].shift(5).round(4)

    # ── Original rolling features ─────────────────────────────────────────────
    g["sentiment_roll3"] = (g["avg_finbert"].shift(1)
                             .rolling(window=3, min_periods=2)
                             .mean().round(4))
    g["sentiment_roll7"] = (g["avg_finbert"].shift(1)
                             .rolling(window=7, min_periods=3)
                             .mean().round(4))

    # ── Original momentum ─────────────────────────────────────────────────────
    g["sentiment_momentum"] = (
        g["avg_finbert"].shift(1) - g["avg_finbert"].shift(3)
    ).round(4)

    # ── NEW: Sentiment acceleration (EDA: trend > level) ─────────────────────
    # Is sentiment momentum itself accelerating or decelerating?
    g["sentiment_acceleration"] = (
        (g["avg_finbert"].shift(1) - g["avg_finbert"].shift(2)) -
        (g["avg_finbert"].shift(2) - g["avg_finbert"].shift(3))
    ).round(4)

    # ── Original volume interaction ───────────────────────────────────────────
    max_articles = g["article_count"].max()
    if max_articles > 0:
        g["sentiment_volume_interaction"] = (
            g["avg_finbert"] * (g["article_count"] / max_articles)
        ).round(4)
    else:
        g["sentiment_volume_interaction"] = 0.0

    # ── NEW: Article count normalized (continuous vs binary) ──────────────────
    # Z-score normalized per ticker — captures relative news volume
    art_mean = g["article_count"].mean()
    art_std  = g["article_count"].std()
    if art_std > 0:
        g["article_count_norm"] = (
            (g["article_count"] - art_mean) / art_std
        ).round(4)
    else:
        g["article_count_norm"] = 0.0

    # ── NEW: Article count change (captures high-volume reversals) ────────────
    # Positive = more articles than yesterday (news picking up)
    # Negative = fewer articles than yesterday (news dying down)
    g["article_count_change"] = g["article_count"].diff().round(4)

    # ── NEW: Volatility x sentiment interaction ───────────────────────────────
    # High volatility + positive sentiment = very different from
    # high volatility + negative sentiment
    g["volatility_sentiment"] = (
        g["volatility_5d"] * g["avg_finbert"]
    ).round(6)

    # ── NEW: Prescore-FinBERT difference (replaces agreement flag) ───────────
    # Positive = FinBERT more bullish than Alpha Vantage
    # Negative = Alpha Vantage more bullish than FinBERT
    # Near zero = both agree on magnitude
    g["prescore_finbert_diff"] = (
        g["avg_finbert"] - g["avg_prescore"]
    ).round(4)

    # ── Original high news day flag (kept alongside norm version) ─────────────
    q75 = g["article_count"].quantile(0.75)
    g["high_news_day"] = (g["article_count"] > q75).astype(int)

    results.append(g)

df = pd.concat(results, ignore_index=True)
print(f"  Sentiment features computed (original + 6 new)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Calendar features
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 6 — Computing calendar features...")

df["day_of_week"]  = df["date"].dt.dayofweek
df["is_month_end"] = df["date"].apply(
    lambda d: 1 if (d + pd.offsets.BDay(1)).month != d.month else 0
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Ticker encoding
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 7 — Encoding tickers...")

ticker_map = {t: i for i, t in enumerate(sorted(df["ticker"].unique()))}
df["ticker_encoded"] = df["ticker"].map(ticker_map)
print(f"  Encoding: {ticker_map}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Target variable
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 8 — Creating target variable...")

df["target"] = (df.groupby("ticker")["close_price"]
                  .shift(-1) > df["close_price"]).astype(float)

last_idx = df.groupby("ticker").tail(1).index
df.loc[last_idx, "target"] = np.nan

up   = (df["target"] == 1).sum()
down = (df["target"] == 0).sum()
print(f"  Target: {int(up):,} UP, {int(down):,} DOWN ({up/(up+down)*100:.1f}% UP)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Final column selection
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Step 9 — Final cleanup...")

df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

FEATURE_COLS_V2 = [
    # Identity
    "ticker", "date", "ticker_encoded",
    # Price features (9)
    "open_price", "high_price", "low_price", "close_price",
    "volume", "log_return", "return_lag1",
    "price_momentum_5d", "volatility_5d", "price_range",
    "rsi_14", "macd", "volume_change",
    # Sentiment features — original (11)
    "avg_finbert", "std_finbert", "bullish_ratio",
    "bearish_ratio",
    "avg_prescore", "article_count",
    "sentiment_lag1", "sentiment_lag2", "sentiment_lag3",
    "sentiment_roll3", "sentiment_roll7",
    "sentiment_momentum",
    "sentiment_volume_interaction",
    "high_news_day",
    # Sentiment features — NEW from EDA (7)
    "sentiment_lag4",
    "sentiment_lag5",
    "sentiment_acceleration",
    "article_count_norm",
    "article_count_change",
    "volatility_sentiment",
    "prescore_finbert_diff",
    # Calendar (2)
    "day_of_week", "is_month_end",
    # Target
    "target"
]

df = df[FEATURE_COLS_V2]

feature_only = [c for c in FEATURE_COLS_V2
                if c not in ["ticker", "date", "target"]]

print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"  Total features: {len(feature_only)} (was 31 in v1)")
print(f"  New features added: 6")
print(f"  Features removed: 1 (prescore_finbert_agreement)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Null report
# ══════════════════════════════════════════════════════════════════════════════
print("\n── NULL REPORT ──────────────────────────────────────────")
nulls = df.isnull().sum()
for col, n in nulls.items():
    if n > 0:
        pct  = n / len(df) * 100
        print(f"  ~ {col:<35} {n:>4} nulls ({pct:.1f}%)")

print(f"  {len(nulls[nulls==0])} columns with zero nulls")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — Compare with v1
# ══════════════════════════════════════════════════════════════════════════════
print("\n── COMPARISON WITH V1 ───────────────────────────────────")
print(f"  v1 features: 31")
print(f"  v2 features: {len(feature_only)}")
print(f"\n  New features added:")
new_features = [
    "sentiment_lag4", "sentiment_lag5", "sentiment_acceleration",
    "article_count_norm", "article_count_change",
    "volatility_sentiment", "prescore_finbert_diff"
]
for f in new_features:
    print(f"    + {f}")
print(f"\n  Features removed:")
print(f"    - prescore_finbert_agreement")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — Save
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/features_v22.csv", index=False)
print(f"\n── SAVED ────────────────────────────────────────────────")
print(f"  data/processed/features_v22.csv")
print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"\n  Original file kept at: data/processed/features.csv")
print("-" * 55)