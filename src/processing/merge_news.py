"""
merge_news.py
─────────────
Merges all cleaned news sources into one master DataFrame.

Sources:
  - alphavantage_clean.csv  (has pre-computed sentiment score)
  - polygon_clean.csv       (no pre-computed score)
  - guardian_clean.csv      (no pre-computed score)

Output:
  - data/processed/news_master.csv
  - data/processed/daily_article_counts.csv

USAGE:
    python src/processing/merge_news.py
"""

import pandas as pd
import numpy as np
import os


# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILES = {
    "AlphaVantage" : "data/processed/alphavantage_clean.csv",
    "Polygon"      : "data/processed/polygon_clean.csv",
    "Guardian"     : "data/processed/guardian_clean.csv",
}

OUTPUT_FILE       = "data/processed/news_master.csv"
DAILY_COUNT_FILE  = "data/processed/daily_article_counts.csv"

FINAL_COLS = [
    "ticker",
    "trading_date",
    "publishedAt",
    "source",
    "title",
    "description",
    "url",
    "text_for_sentiment",
    "data_source",
    "prescore",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def normalize_to_standard_scale(series: pd.Series) -> pd.Series:
    """Min-max normalize to [-1, +1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series.fillna(0.0)
    return (2 * (series - mn) / (mx - mn) - 1).round(4)


# ══════════════════════════════════════════════════════════════════════════════
# MERGE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def merge_news_sources(
    input_files : dict = INPUT_FILES,
    output_file : str  = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Merges all cleaned news sources into one master DataFrame.
    Returns merged DataFrame.
    """

    print("── MERGING NEWS SOURCES ─────────────────────────────────")

    # ── 1. Load all sources ───────────────────────────────────────────────────
    frames = {}
    for name, path in input_files.items():
        if not os.path.exists(path):
            print(f"  ⚠ {name}: file not found at {path} — skipping")
            continue
        df = pd.read_csv(path)
        if "data_source" not in df.columns:
            df["data_source"] = name
        frames[name] = df
        print(f"  Loaded {name:<15} : {len(df):>6,} rows")

    if not frames:
        print("  ✖ No files found to merge")
        return pd.DataFrame()

    # ── 2. Normalize pre-computed scores → prescore ───────────────────────────
    print(f"\n  ── Score normalization ──")

    if "AlphaVantage" in frames:
        av     = frames["AlphaVantage"]
        raw_av = av["alphavantage_score"]
        av["prescore"] = normalize_to_standard_scale(raw_av)
        print(f"  AlphaVantage raw  → min: {raw_av.min():.4f}  max: {raw_av.max():.4f}  mean: {raw_av.mean():.4f}")
        print(f"  AlphaVantage norm → min: {av['prescore'].min():.4f}  max: {av['prescore'].max():.4f}  mean: {av['prescore'].mean():.4f}")
        av = av.drop(columns=["alphavantage_score", "alphavantage_label"], errors="ignore")
        frames["AlphaVantage"] = av

    if "Polygon" in frames:
        frames["Polygon"]["prescore"] = np.nan

    if "Guardian" in frames:
        frames["Guardian"]["prescore"] = np.nan

    # ── 3. Align columns ──────────────────────────────────────────────────────
    for name in list(frames.keys()):
        for col in FINAL_COLS:
            if col not in frames[name].columns:
                frames[name][col] = np.nan
        frames[name] = frames[name][FINAL_COLS]

    # ── 4. Concatenate ────────────────────────────────────────────────────────
    df = pd.concat(frames.values(), ignore_index=True)
    print(f"\n✓ Combined : {len(df):,} rows × {df.shape[1]} cols")

    # ── 5. Fix dtypes ─────────────────────────────────────────────────────────
    df["trading_date"] = pd.to_datetime(df["trading_date"], utc=True)
    df["publishedAt"]  = pd.to_datetime(df["publishedAt"],  utc=True)
    print(f"✓ Dtypes fixed")

    # ── 6. Remove cross-source duplicate URLs ─────────────────────────────────
    before = len(df)
    df["has_prescore"] = df["prescore"].notna().astype(int)
    df = df.sort_values("has_prescore", ascending=False)
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop(columns=["has_prescore"])
    print(f"✓ Removed {before - len(df):,} cross-source duplicate URLs")

    # ── 7. Remove cross-source duplicate titles ───────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "title"], keep="first")
    print(f"✓ Removed {before - len(df):,} cross-source duplicate titles")

    # ── 8. Sort ───────────────────────────────────────────────────────────────
    df = df.sort_values(["ticker", "trading_date", "publishedAt"]).reset_index(drop=True)
    print(f"✓ Sorted by ticker → trading_date → publishedAt")

    # ── 9. Null check ─────────────────────────────────────────────────────────
    print(f"\n  Final null counts:")
    core = ["ticker", "trading_date", "title", "url", "text_for_sentiment", "data_source"]
    for col, n in df.isnull().sum().items():
        pct = n / len(df) * 100
        if col in core and n > 0:
            print(f"  ⚠ {col:<25} {n:,} nulls ({pct:.1f}%) <- PROBLEM")
        elif n > 0:
            print(f"  ~ {col:<25} {n:,} nulls ({pct:.1f}%) <- expected")
        else:
            print(f"  ✓ {col:<25} 0 nulls")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print(f"\n── MERGE COMPLETE ───────────────────────────────────────")
    print(f"  Total articles : {len(df):,}")
    print(f"  Columns        : {list(df.columns)}")

    print(f"\n  Per-source breakdown:")
    for src, grp in df.groupby("data_source"):
        pct = len(grp) / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f"    {src:<15} {len(grp):>6,} ({pct:4.1f}%) {bar}")

    print(f"\n  Per-ticker breakdown:")
    for ticker, grp in df.groupby("ticker"):
        has_pre = grp["prescore"].notna().sum()
        pct_pre = has_pre / len(grp) * 100
        print(f"\n    {ticker}:")
        print(f"      Total articles : {len(grp):,}")
        print(f"      Date range     : {grp['trading_date'].min().date()} → {grp['trading_date'].max().date()}")
        print(f"      With prescore  : {has_pre:,} ({pct_pre:.1f}%)")
        print(f"      By source      :")
        for src, cnt in grp["data_source"].value_counts().items():
            print(f"        {src:<15} {cnt:>5,}")

    print(f"\n  Daily article count (last 5 trading days):")
    daily = (df.groupby(["trading_date", "ticker"]).size().unstack(fill_value=0))
    print(daily.tail(5).to_string())
    print("─" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} cols")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# DAILY ARTICLE COUNT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def daily_article_counts(
    df          : pd.DataFrame,
    output_file : str = DAILY_COUNT_FILE,
) -> pd.DataFrame:
    """
    Computes article count per ticker per trading day.
    Saves result to output_file.

    Args:
        df          : merged news_master DataFrame
        output_file : path to save daily counts CSV

    Returns:
        DataFrame with columns: ticker, trading_date, article_count
    """

    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], utc=True)

    daily = (df.groupby(["ticker", "trading_date"])
               .size()
               .reset_index(name="article_count")
               .sort_values(["ticker", "trading_date"])
               .reset_index(drop=True))

    print("\n── DAILY ARTICLE COUNT PER TICKER ───────────────────────")

    print(f"\n  Overall stats:")
    print(f"    Total ticker-days     : {len(daily):,}")
    print(f"    Avg articles per day  : {daily['article_count'].mean():.1f}")
    print(f"    Max articles per day  : {daily['article_count'].max()}")
    print(f"    Min articles per day  : {daily['article_count'].min()}")

    print(f"\n  Per-ticker daily stats:")
    for ticker, grp in daily.groupby("ticker"):
        print(f"\n  {ticker}:")
        print(f"    Days with articles : {len(grp)}")
        print(f"    Avg per day        : {grp['article_count'].mean():.1f}")
        print(f"    Max per day        : {grp['article_count'].max()}")
        print(f"    Min per day        : {grp['article_count'].min()}")

    print(f"\n  Top 10 highest volume days:")
    top = daily.nlargest(10, "article_count")
    print(top[["ticker", "trading_date", "article_count"]].to_string(index=False))

    print(f"\n  Bottom 10 lowest volume days:")
    bottom = daily.nsmallest(10, "article_count")
    print(bottom[["ticker", "trading_date", "article_count"]].to_string(index=False))

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    daily.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")

    return daily

def missing_coverage_report(
    df          : pd.DataFrame,
    start_date  : str = "2024-05-01",
    end_date    : str = "2026-05-29",
    output_file : str = "data/processed/missing_coverage.csv",
) -> pd.DataFrame:
    """
    Finds all trading days where a ticker had zero articles.
    Compares actual coverage against all trading days in the date range.

    Args:
        df         : merged news_master DataFrame
        start_date : start of expected date range
        end_date   : end of expected date range
        output_file: path to save missing days CSV

    Returns:
        DataFrame with columns: ticker, missing_date
    """

    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], utc=True).dt.date

    # Generate all trading days in range (excludes weekends + US holidays)
    all_trading_days = pd.bdate_range(start=start_date, end=end_date).date

    tickers = sorted(df["ticker"].unique().tolist())
    missing_rows = []

    print("\n── MISSING COVERAGE REPORT ──────────────────────────────")
    print(f"  Date range    : {start_date} → {end_date}")
    print(f"  Trading days  : {len(all_trading_days)}")
    print(f"  Tickers       : {tickers}")

    for ticker in tickers:
        # Days that have at least one article
        covered = set(
            df[df["ticker"] == ticker]["trading_date"].unique()
        )

        # Days with no articles
        missing = [d for d in all_trading_days if d not in covered]

        for day in missing:
            missing_rows.append({
                "ticker"       : ticker,
                "missing_date" : day,
            })

        coverage_pct = (len(all_trading_days) - len(missing)) / len(all_trading_days) * 100
        print(f"\n  {ticker}:")
        print(f"    Covered days : {len(all_trading_days) - len(missing)}/{len(all_trading_days)} ({coverage_pct:.1f}%)")
        print(f"    Missing days : {len(missing)}")
        if missing:
            # Show first and last 3 missing days
            show = missing[:3] + (["..."] if len(missing) > 6 else []) + (missing[-3:] if len(missing) > 3 else [])
            for d in show:
                print(f"      {d}")

    result = pd.DataFrame(missing_rows)

    print(f"\n  Total missing ticker-days : {len(result):,}")
    print(f"  Total possible ticker-days: {len(tickers) * len(all_trading_days):,}")
    overall_coverage = (1 - len(result) / (len(tickers) * len(all_trading_days))) * 100
    print(f"  Overall coverage          : {overall_coverage:.1f}%")

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")

    return result


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_master    = merge_news_sources()
    daily_counts = daily_article_counts(df_master)
    missing      = missing_coverage_report(df_master)
