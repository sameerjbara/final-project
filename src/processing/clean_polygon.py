"""
clean_polygon.py
────────────────
Cleans merged Polygon.io news data.
Works for any date range and any number of tickers.

Input  : data/raw/polygon_raw.csv
Output : data/processed/polygon_clean.csv

Steps:
  1. Drop unused columns
  2. Parse publishedAt (ISO UTC format)
  3. Trading day alignment (articles after 4pm ET → next trading day)
  4. Drop missing titles
  5. Build text_for_sentiment
  6. Deduplicate by URL
  7. Deduplicate by ticker + title
  8. Relevance filter (ticker must appear in title or description)
  9. Drop very short texts
  10. Final null check + summary
"""

import pandas as pd
import numpy as np


# ── Relevance keywords per ticker ─────────────────────────────────────────────
RELEVANCE = {
    "AAPL"  : ["apple", "aapl", "iphone", "ipad", "macbook", "tim cook",
                "app store", "ios", "macos", "airpods", "vision pro"],
    "NVDA"  : ["nvidia", "nvda", "jensen huang", "geforce", "cuda",
                "h100", "blackwell", "gpu", "rtx", "hopper"],
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
    "NFLX"  : ["netflix", "nflx", "streaming", "reed hastings"],
}


def is_relevant(row: pd.Series) -> bool:
    """Returns True if the article is relevant to the ticker."""
    text     = (str(row["title"]) + " " +
                str(row["description"])).lower()
    keywords = RELEVANCE.get(row["ticker"], [])
    return any(kw in text for kw in keywords)


def clean_polygon(
    input_path  : str = "data/raw/polygon_raw.csv",
    output_path : str = "data/processed/polygon_clean.csv",
) -> pd.DataFrame:
    """
    Cleans Polygon news data and saves to output_path.
    Returns cleaned DataFrame.
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print("── STARTING POLYGON CLEANING ────────────────────────────")
    print(f"  Input  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Tickers: {sorted(df['ticker'].unique().tolist())}")

    # ── 1. Drop unused columns ────────────────────────────────────────────────
    drop_cols = [c for c in ["company", "date"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"\n✓ Dropped unused columns: {drop_cols}")

    # ── 2. Parse publishedAt ──────────────────────────────────────────────────
    # Polygon format: 2024-05-01T04:53:12Z (ISO 8601 UTC)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)

    before = len(df)
    df = df.dropna(subset=["publishedAt"])
    if before - len(df) > 0:
        print(f"  ⚠ Dropped {before - len(df)} rows with unparseable dates")

    print(f"✓ Dates parsed (UTC)")
    print(f"  Range: {df['publishedAt'].min()} → {df['publishedAt'].max()}")

    # ── 3. Trading day alignment ──────────────────────────────────────────────
    # Articles published after 4pm ET (21:00 UTC) → next trading day
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
    # description and content are identical in Polygon — use description
    df["text_for_sentiment"] = (
        df["title"].fillna("") + ". " + df["description"].fillna("")
    ).str.strip(". ")
    print(f"✓ Created text_for_sentiment")

    # ── 6. Deduplicate by URL ─────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"✓ Dropped {before - len(df):,} duplicate URLs")

    # ── 7. Deduplicate by ticker + title ──────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "title"])
    print(f"✓ Dropped {before - len(df):,} duplicate titles")

    # ── 8. Relevance filter ───────────────────────────────────────────────────
    # Polygon tags articles with multiple tickers — filter for actual relevance
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

    # ── 9. Drop very short texts ──────────────────────────────────────────────
    before = len(df)
    df = df[df["text_for_sentiment"].str.split().str.len() >= 5]
    print(f"✓ Dropped {before - len(df):,} articles under 5 words")

    # ── 10. Add data_source ───────────────────────────────────────────────────
    df["data_source"] = "Polygon"

    # ── 11. Select final columns ──────────────────────────────────────────────
    df = df[[
        "ticker",
        "trading_date",
        "publishedAt",
        "source",
        "title",
        "description",
        "url",
        "text_for_sentiment",
        "data_source",
    ]].reset_index(drop=True)

    # ── 12. Final null check ──────────────────────────────────────────────────
    print(f"\n  Final null counts:")
    for col, n in df.isnull().sum().items():
        flag = "⚠" if n > 0 and col not in ["description"] else "✓"
        print(f"  {flag} {col:<30} {n:,} nulls")

    # ── 13. Summary ───────────────────────────────────────────────────────────
    print(f"\n── CLEANING COMPLETE ────────────────────────────────────")
    print(f"  Output : {len(df):,} rows × {df.shape[1]} cols")
    print(f"\n  Per-ticker summary:")
    for ticker, grp in df.groupby("ticker"):
        print(f"\n  {ticker}:")
        print(f"    Articles       : {len(grp):,}")
        print(f"    Date range     : "
              f"{grp['trading_date'].min().date()} → "
              f"{grp['trading_date'].max().date()}")
        print(f"    Unique sources : {grp['source'].nunique()}")
        print(f"    Top sources    :")
        for src, cnt in grp["source"].value_counts().head(3).items():
            pct = cnt / len(grp) * 100
            print(f"      {src:<30} {cnt:>4} ({pct:.1f}%)")

    print(f"\n  Final columns: {list(df.columns)}")
    print("─" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

    return df


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_clean = clean_polygon(
        input_path  = "data/raw/polygon_raw.csv",
        output_path = "data/processed/polygon_clean.csv",
    )

    print("\nFirst 3 rows:")
    print(df_clean[["ticker", "trading_date", "source",
                     "title"]].head(3).to_string(index=False))