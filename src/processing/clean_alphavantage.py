"""
clean_alphavantage.py
─────────────────────
Cleans merged Alpha Vantage news data.
Works for any date range and any number of tickers.

Input  : data/raw/alphavantage_raw.csv
Output : data/processed/alphavantage_clean.csv

Steps:
  1. Drop unused columns
  2. Parse publishedAt (YYYYMMDDTHHMMSS format)
  3. Trading day alignment
  4. Drop missing titles
  5. Build text_for_sentiment
  6. Normalize sentiment score to [-1, +1]
  7. Deduplicate by URL
  8. Deduplicate by ticker + title
  9. Relevance filter
  10. Drop very short texts
  11. Final null check + summary
"""

import pandas as pd
import numpy as np
import os


# ── Relevance keywords per ticker ─────────────────────────────────────────────
RELEVANCE = {
    "AAPL"  : ["apple", "aapl", "iphone", "ipad", "macbook", "tim cook",
                "app store", "ios", "macos", "airpods", "vision pro"],
    "NVDA"  : ["nvidia", "nvda", "jensen huang", "geforce", "cuda",
                "h100", "blackwell", "gpu", "rtx"],
    "MSFT"  : ["microsoft", "msft", "azure", "copilot", "satya nadella",
                "windows", "xbox", "teams", "openai"],
    "AMZN"  : ["amazon", "amzn", "aws", "andy jassy", "prime",
                "alexa", "kindle", "whole foods"],
    "GOOGL" : ["google", "googl", "alphabet", "sundar pichai", "gemini",
                "youtube", "waymo", "deepmind", "chrome"],
    "META"  : ["meta", "facebook", "instagram", "whatsapp", "mark zuckerberg",
                "zuckerberg", "threads", "oculus", "llama"],
    "JPM"   : ["jpmorgan", "jpm", "jamie dimon", "chase", "j.p. morgan",
                "jpmc"],
    "NFLX"  : ["netflix", "nflx", "streaming", "reed hastings",
                "netflix stock", "netflix earnings", "netflix subscribers"],
}


def normalize_to_standard_scale(series: pd.Series) -> pd.Series:
    """Min-max normalize to [-1, +1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series.fillna(0.0)
    return 2 * (series - mn) / (mx - mn) - 1


def is_relevant(row: pd.Series) -> bool:
    """Returns True if the article is relevant to the ticker."""
    text     = (str(row["title"]) + " " +
                str(row["description"])).lower()
    keywords = RELEVANCE.get(row["ticker"], [])
    return any(kw in text for kw in keywords)


def clean_alphavantage(
    input_path  : str = "data/raw/alphavantage_raw.csv",
    output_path : str = "data/processed/alphavantage_clean.csv",
) -> pd.DataFrame:
    """
    Cleans Alpha Vantage news data and saves to output_path.
    Returns cleaned DataFrame.
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print("── STARTING ALPHA VANTAGE CLEANING ──────────────────────")
    print(f"  Input  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Tickers: {sorted(df['ticker'].unique().tolist())}")

    # ── 1. Drop unused columns ────────────────────────────────────────────────
    drop_cols = [c for c in ["company", "date", "content"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"\n✓ Dropped unused columns: {drop_cols}")

    # ── 2. Parse publishedAt ──────────────────────────────────────────────────
    # Alpha Vantage format: 20240531T141602 (no separators, no timezone)
    def parse_av_date(val):
        try:
            return pd.to_datetime(str(val), format="%Y%m%dT%H%M%S", utc=True)
        except Exception:
            try:
                return pd.to_datetime(str(val), utc=True)
            except Exception:
                return pd.NaT

    df["publishedAt"] = df["publishedAt"].apply(parse_av_date)

    before = len(df)
    df = df.dropna(subset=["publishedAt"])
    if before - len(df) > 0:
        print(f"  ⚠ Dropped {before - len(df)} rows with unparseable dates")

    print(f"✓ Dates parsed (UTC)")
    print(f"  Range: {df['publishedAt'].min()} → {df['publishedAt'].max()}")

    # ── 3. Trading day alignment ──────────────────────────────────────────────
    def to_trading_day(ts):
        if pd.isna(ts):
            return pd.NaT
        if ts.hour >= 21:
            ts = ts + pd.offsets.BDay(1)
        if ts.weekday() >= 5:
            ts = ts + pd.offsets.BDay(1)
        return ts.normalize()

    df["trading_date"] = df["publishedAt"].apply(to_trading_day)
    shifted = (df["trading_date"].dt.date != df["publishedAt"].dt.date).sum()
    print(f"✓ Trading day alignment — {shifted:,} articles shifted")

    # ── 4. Drop missing titles ────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]
    print(f"✓ Dropped {before - len(df)} rows with missing title")

    # ── 5. Build text_for_sentiment ───────────────────────────────────────────
    df["text_for_sentiment"] = (
        df["title"].fillna("") + ". " + df["description"].fillna("")
    ).str.strip(". ")
    print(f"✓ Created text_for_sentiment")

    # ── 6. Normalize sentiment score ──────────────────────────────────────────
    score_col = "overall_sentiment_score"
    label_col = "overall_sentiment_label"

    print(f"\n  Raw sentiment_score stats:")
    print(f"    Mean : {df[score_col].mean():.4f}")
    print(f"    Std  : {df[score_col].std():.4f}")
    print(f"    Min  : {df[score_col].min():.4f}")
    print(f"    Max  : {df[score_col].max():.4f}")

    df[score_col] = df[score_col].clip(-1, 1)
    df["alphavantage_score"] = normalize_to_standard_scale(
        df[score_col]
    ).round(4)

    print(f"\n  After normalization:")
    print(f"    Mean : {df['alphavantage_score'].mean():.4f}")
    print(f"    Min  : {df['alphavantage_score'].min():.4f}")
    print(f"    Max  : {df['alphavantage_score'].max():.4f}")

    print(f"\n  Sentiment label distribution:")
    for label, cnt in df[label_col].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {label:<25} {cnt:>6,} ({pct:.1f}%)")

    df = df.drop(columns=[score_col])
    df = df.rename(columns={label_col: "alphavantage_label"})
    print(f"\n✓ Scores normalized → alphavantage_score")
    print(f"✓ Label renamed → alphavantage_label")

    # ── 7. Deduplicate by URL ─────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"✓ Dropped {before - len(df):,} duplicate URLs")

    # ── 8. Deduplicate by ticker + title ──────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "title"])
    print(f"✓ Dropped {before - len(df):,} duplicate titles")

    # ── 9. Relevance filter ───────────────────────────────────────────────────
    before = len(df)
    df["is_relevant"] = df.apply(is_relevant, axis=1)
    irrelevant = df[~df["is_relevant"]]

    print(f"\n  ── Relevance filter ──")
    print(f"  Irrelevant articles : {len(irrelevant):,}")
    if len(irrelevant) > 0:
        print(f"  Sample irrelevant titles:")
        for title in irrelevant["title"].head(3).values:
            print(f"    • {title[:80]}")

    df = df[df["is_relevant"]].drop(columns=["is_relevant"])
    print(f"✓ Kept {len(df):,} relevant articles "
          f"(dropped {before - len(df):,})")

    # ── 10. Drop very short texts ─────────────────────────────────────────────
    before = len(df)
    df = df[df["text_for_sentiment"].str.split().str.len() >= 5]
    print(f"✓ Dropped {before - len(df):,} articles under 5 words")

    # ── 11. Add data_source ───────────────────────────────────────────────────
    df["data_source"] = "AlphaVantage"

    # ── 12. Select final columns ──────────────────────────────────────────────
    df = df[[
        "ticker",
        "trading_date",
        "publishedAt",
        "source",
        "title",
        "description",
        "url",
        "text_for_sentiment",
        "alphavantage_score",
        "alphavantage_label",
        "data_source",
    ]].reset_index(drop=True)

    # ── 13. Final null check ──────────────────────────────────────────────────
    print(f"\n  Final null counts:")
    for col, n in df.isnull().sum().items():
        flag = "⚠" if n > 0 and col not in ["description"] else "✓"
        print(f"  {flag} {col:<30} {n:,} nulls")

    # ── 14. Summary ───────────────────────────────────────────────────────────
    print(f"\n── CLEANING COMPLETE ────────────────────────────────────")
    print(f"  Output : {len(df):,} rows × {df.shape[1]} cols")
    print(f"\n  Per-ticker summary:")
    for ticker, grp in df.groupby("ticker"):
        print(f"\n  {ticker}:")
        print(f"    Articles         : {len(grp):,}")
        print(f"    Date range       : "
              f"{grp['trading_date'].min().date()} → "
              f"{grp['trading_date'].max().date()}")
        print(f"    Avg AV score     : {grp['alphavantage_score'].mean():.4f}")
        print(f"    Unique sources   : {grp['source'].nunique()}")
        print(f"    Label breakdown  :")
        for label, cnt in grp["alphavantage_label"].value_counts().items():
            pct = cnt / len(grp) * 100
            print(f"      {label:<25} {cnt:>4} ({pct:.1f}%)")

    print(f"\n  Final columns: {list(df.columns)}")
    print("─" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

    return df


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_clean = clean_alphavantage(
        input_path  = "data/raw/alphavantage_raw.csv",
        output_path = "data/processed/alphavantage_clean.csv",
    )

    print("\nFirst 3 rows:")
    print(df_clean[["ticker", "trading_date", "source",
                     "alphavantage_score", "title"]].head(3).to_string(index=False))