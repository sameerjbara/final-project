"""
model_per_ticker.py
───────────────────
Trains a separate XGBoost + LightGBM + SVM ensemble per ticker.

Runs TWO models per ticker:
  1. Full model    — price + sentiment features (37 features)
  2. Price only    — price features only (10 features)

Compares both to show how much sentiment adds over price alone.

Output:
  models/ticker_results_v2.csv     ← full model results
  models/ticker_results_v2.png     ← full model chart
  models/ticker_results_price.csv  ← price only results
  models/ablation_comparison.csv   ← side by side comparison
  models/ablation_comparison.png   ← comparison chart

USAGE:
    python src/models/model_per_ticker.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm           import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (accuracy_score, f1_score,
                                   roc_auc_score, confusion_matrix)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH         = "data/processed/features_v22.csv"
RESULTS_PATH_FULL = "models/ticker_results_v2.png"
CSV_PATH_FULL     = "models/ticker_results_v2.csv"
RESULTS_PATH_PRCE = "models/ticker_results_price.png"
CSV_PATH_PRCE     = "models/ticker_results_price.csv"
ABLATION_CSV      = "models/ablation_comparison.csv"
ABLATION_PNG      = "models/ablation_comparison.png"

# ── Feature sets ──────────────────────────────────────────────────────────────
PRICE_FEATURES = [
    "log_return", "return_lag1", "price_momentum_5d",
    "volatility_5d", "price_range", "rsi_14", "macd",
    "volume_change", "day_of_week", "is_month_end",
]

SENTIMENT_FEATURES = [
    "avg_finbert", "std_finbert", "bullish_ratio",
    "bearish_ratio", "avg_prescore", "article_count",
    "sentiment_lag1", "sentiment_lag2", "sentiment_lag3",
    "sentiment_roll3", "sentiment_roll7", "sentiment_momentum",
    "sentiment_volume_interaction", "high_news_day",
    "sentiment_lag4", "sentiment_lag5", "sentiment_acceleration",
    "article_count_norm", "article_count_change",
    "volatility_sentiment", "prescore_finbert_diff",
]

FULL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES

print("── PER-TICKER MODELING ──────────────────────────────────")
print(f"  Price features    : {len(PRICE_FEATURES)}")
print(f"  Sentiment features: {len(SENTIMENT_FEATURES)}")
print(f"  Full features     : {len(FULL_FEATURES)}")
print(f"  Models            : XGBoost + LightGBM + SVM ensemble")
print(f"  Split             : 75% train / 15% val / 10% test")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 1: Loading data ─────────────────────────────────")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
df = df.dropna(subset=FULL_FEATURES + ["target"])

print(f"  Loaded {len(df):,} usable rows")
print(f"  Tickers: {sorted(df['ticker'].unique().tolist())}")

dates     = sorted(df["date"].unique())
n         = len(dates)
train_end = dates[int(n * 0.75)]
val_end   = dates[int(n * 0.90)]

print(f"\n  Train end : {train_end.date()}")
print(f"  Val end   : {val_end.date()}")
print(f"  Test start: {dates[int(n * 0.90) + 1].date()}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — TRAIN AND EVALUATE ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def train_evaluate(train, val, test, features, label):
    """
    Train XGBoost + LightGBM + SVM ensemble on given feature set.
    Returns dict with accuracy, AUC, F1, top feature.
    """
    X_trainval = np.vstack([
        StandardScaler().fit(train[features]).transform(train[features]),
        StandardScaler().fit(train[features]).transform(val[features])
    ])

    scaler     = StandardScaler()
    X_train    = scaler.fit_transform(train[features])
    X_val      = scaler.transform(val[features])
    X_test     = scaler.transform(test[features])
    X_trainval = np.vstack([X_train, X_val])

    y_train    = train["target"].values
    y_val      = val["target"].values
    y_test     = test["target"].values
    y_trainval = np.concatenate([y_train, y_val])

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, eval_metric="auc",
        early_stopping_rounds=30, verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_f = XGBClassifier(
        n_estimators=max(xgb.best_iteration + 1, 10),
        max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, verbosity=0,
    )
    xgb_f.fit(X_trainval, y_trainval)

    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbose=-1,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(period=-1)])

    lgbm_f = LGBMClassifier(
        n_estimators=max(lgbm.best_iteration_, 10),
        max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbose=-1,
    )
    lgbm_f.fit(X_trainval, y_trainval)

    # SVM
    svm = SVC(kernel="rbf", C=1.0, gamma="scale",
              probability=True, random_state=SEED)
    svm.fit(X_trainval, y_trainval)

    # Ensemble
    prob = (0.4 * xgb_f.predict_proba(X_test)[:, 1] +
            0.4 * lgbm_f.predict_proba(X_test)[:, 1] +
            0.2 * svm.predict_proba(X_test)[:, 1])
    pred = (prob > 0.5).astype(int)

    acc = accuracy_score(y_test, pred) * 100
    auc = roc_auc_score(y_test, prob)  * 100
    f1  = f1_score(y_test, pred, zero_division=0) * 100
    cm  = confusion_matrix(y_test, pred)

    xgb_imp  = pd.Series(
        xgb_f.feature_importances_, index=features
    ).sort_values(ascending=False)

    xgb_acc  = accuracy_score(y_test, (xgb_f.predict_proba(X_test)[:,1]>0.5).astype(int)) * 100
    lgbm_acc = accuracy_score(y_test, (lgbm_f.predict_proba(X_test)[:,1]>0.5).astype(int)) * 100
    svm_acc  = accuracy_score(y_test, (svm.predict_proba(X_test)[:,1]>0.5).astype(int)) * 100

    print(f"    [{label}] XGB={xgb_acc:.1f}% LGBM={lgbm_acc:.1f}% "
          f"SVM={svm_acc:.1f}% → Ensemble={acc:.1f}% AUC={auc:.1f}%")
    print(f"    [{label}] CM: TN={cm[0,0]} FP={cm[0,1]} "
          f"FN={cm[1,0]} TP={cm[1,1]}")
    print(f"    [{label}] Top feature: {xgb_imp.index[0]}")

    return {
        "accuracy"   : acc,
        "auc"        : auc,
        "f1"         : f1,
        "xgb_acc"    : xgb_acc,
        "lgbm_acc"   : lgbm_acc,
        "svm_acc"    : svm_acc,
        "top_feature": xgb_imp.index[0],
        "cm_tn"      : cm[0, 0],
        "cm_fp"      : cm[0, 1],
        "cm_fn"      : cm[1, 0],
        "cm_tp"      : cm[1, 1],
        "test_rows"  : len(test),
        "test_up_pct": test["target"].mean() * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train per ticker — both full and price only
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 2: Training per-ticker models ───────────────────")

full_results  = []
price_results = []

for ticker in sorted(df["ticker"].unique()):
    print(f"\n  ── {ticker} ──────────────────────────────────────────")

    df_t  = df[df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    train = df_t[df_t["date"] <= train_end].reset_index(drop=True)
    val   = df_t[(df_t["date"] > train_end) & (df_t["date"] <= val_end)].reset_index(drop=True)
    test  = df_t[df_t["date"] > val_end].reset_index(drop=True)

    print(f"    Train: {len(train)}  Val: {len(val)}  Test: {len(test)}  "
          f"UP%: {test['target'].mean()*100:.1f}%")

    if len(test) < 20:
        print(f"    ⚠ Skipping")
        continue

    # Full model — price + sentiment
    full = train_evaluate(train, val, test, FULL_FEATURES, "FULL")
    full["ticker"] = ticker
    full_results.append(full)

    # Price only model — no sentiment
    price = train_evaluate(train, val, test, PRICE_FEATURES, "PRICE")
    price["ticker"] = ticker
    price_results.append(price)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 3: Results summary ──────────────────────────────")

full_df  = pd.DataFrame(full_results).sort_values("accuracy", ascending=False)
price_df = pd.DataFrame(price_results)
price_df = price_df.set_index("ticker").reindex(full_df["ticker"]).reset_index()

print(f"\n  {'Ticker':<8} {'Full acc':>10} {'Price acc':>10} "
      f"{'Sent. gain':>12} {'Full AUC':>10} {'Top feature'}")
print(f"  {'-'*75}")

ablation_rows = []
for _, frow in full_df.iterrows():
    ticker    = frow["ticker"]
    prow      = price_df[price_df["ticker"]==ticker].iloc[0]
    gain      = frow["accuracy"] - prow["accuracy"]
    marker    = "✓" if gain > 0 else "✗"
    print(f"  {ticker:<8} {frow['accuracy']:>9.1f}% "
          f"{prow['accuracy']:>9.1f}% "
          f"{gain:>+10.1f}% {marker} "
          f"{frow['auc']:>9.1f}%  "
          f"{frow['top_feature']}")
    ablation_rows.append({
        "ticker"         : ticker,
        "full_acc"       : frow["accuracy"],
        "full_auc"       : frow["auc"],
        "full_f1"        : frow["f1"],
        "price_acc"      : prow["accuracy"],
        "price_auc"      : prow["auc"],
        "sentiment_gain" : gain,
        "top_feature_full" : frow["top_feature"],
        "top_feature_price": prow["top_feature"],
        "test_rows"      : int(frow["test_rows"]),
        "test_up_pct"    : frow["test_up_pct"],
    })

ablation_df = pd.DataFrame(ablation_rows)

avg_full  = full_df["accuracy"].mean()
avg_price = price_df["accuracy"].mean()
avg_gain  = ablation_df["sentiment_gain"].mean()

print(f"  {'-'*75}")
print(f"  {'Average':<8} {avg_full:>9.1f}% {avg_price:>9.1f}% "
      f"{avg_gain:>+10.1f}%")
print(f"\n  Full model avg    : {avg_full:.1f}%")
print(f"  Price only avg    : {avg_price:.1f}%")
print(f"  Sentiment adds    : {avg_gain:+.1f}% on average")
print(f"  Sentiment helps   : {(ablation_df['sentiment_gain']>0).sum()}/6 tickers")
print(f"  Sentiment hurts   : {(ablation_df['sentiment_gain']<=0).sum()}/6 tickers")
print(f"\n  Full model best   : {full_df.iloc[0]['ticker']} "
      f"({full_df.iloc[0]['accuracy']:.1f}%)")
print(f"  Full model worst  : {full_df.iloc[-1]['ticker']} "
      f"({full_df.iloc[-1]['accuracy']:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Charts
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 4: Saving charts ────────────────────────────────")
os.makedirs("models", exist_ok=True)

# Chart 1 — Full model accuracy + AUC
fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 6)))
fig.patch.set_facecolor("#FAFAFA")

ax = axes[0]
colors = ["#2E7D32" if a>=60 else "#F9A825" if a>=55 else "#C62828"
          for a in full_df["accuracy"]]
bars = ax.barh(full_df["ticker"], full_df["accuracy"],
               color=colors, alpha=0.85, edgecolor="white")
ax.axvline(x=50, color="black", linewidth=1.5, linestyle="--",
           alpha=0.5, label="Random (50%)")
ax.axvline(x=55, color="#F9A825", linewidth=1.5, linestyle=":",
           alpha=0.7, label="Good (55%)")
ax.axvline(x=60, color="#2E7D32", linewidth=1.5, linestyle=":",
           alpha=0.7, label="Target (60%)")
ax.set_xlabel("Accuracy (%)")
ax.set_title("Full model accuracy per ticker\n(price + sentiment features)",
             fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(40, 80)
for bar, val in zip(bars, full_df["accuracy"]):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

ax = axes[1]
colors_auc = ["#2E7D32" if a>=60 else "#F9A825" if a>=55 else "#C62828"
              for a in full_df["auc"]]
bars2 = ax.barh(full_df["ticker"], full_df["auc"],
                color=colors_auc, alpha=0.85, edgecolor="white")
ax.axvline(x=50, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
ax.axvline(x=55, color="#F9A825", linewidth=1.5, linestyle=":", alpha=0.7)
ax.axvline(x=60, color="#2E7D32", linewidth=1.5, linestyle=":", alpha=0.7)
ax.set_xlabel("ROC-AUC (%)")
ax.set_title("Full model AUC per ticker",
             fontweight="bold")
ax.set_xlim(40, 80)
for bar, val in zip(bars2, full_df["auc"]):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

fig.suptitle("Per-Ticker Model Results — Price + Sentiment Features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_PATH_FULL, dpi=150, bbox_inches="tight",
            facecolor="#FAFAFA")
plt.close()
print(f"  Saved: {RESULTS_PATH_FULL}")

# Chart 2 — Ablation comparison
tickers_ab = ablation_df["ticker"].tolist()
x = np.arange(len(tickers_ab))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor("#FAFAFA")

ax = axes[0]
b1 = ax.barh(x+w/2, ablation_df["price_acc"], w,
             label=f"Price only (avg {avg_price:.1f}%)",
             color="#1565C0", alpha=0.85)
b2 = ax.barh(x-w/2, ablation_df["full_acc"],  w,
             label=f"Full model (avg {avg_full:.1f}%)",
             color="#B71C1C", alpha=0.85)
ax.axvline(x=50, color="black", linewidth=2, linestyle="--",
           alpha=0.6, label="Random (50%)")
ax.set_yticks(x)
ax.set_yticklabels(tickers_ab)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Price only vs Full model per ticker\n"
             "How much does sentiment add?",
             fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(35, 85)
for bar, val in zip(b1, ablation_df["price_acc"]):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9,
            color="#1565C0", fontweight="bold")
for bar, val in zip(b2, ablation_df["full_acc"]):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9,
            color="#B71C1C", fontweight="bold")

ax = axes[1]
gains  = ablation_df["sentiment_gain"].tolist()
colors = ["#2E7D32" if g>0 else "#C62828" for g in gains]
bars3  = ax.barh(tickers_ab, gains, color=colors,
                 alpha=0.85, edgecolor="white")
ax.axvline(x=0, color="black", linewidth=2, alpha=0.7)
ax.set_xlabel("Accuracy gain from adding sentiment (%)")
ax.set_title(f"Sentiment contribution per ticker\n"
             f"(Full − Price only) | avg gain: {avg_gain:+.1f}%\n"
             f"Green = sentiment helps, Red = sentiment hurts",
             fontweight="bold")
for bar, val in zip(bars3, gains):
    ax.text(val+(0.3 if val>=0 else -0.3),
            bar.get_y()+bar.get_height()/2,
            f"{val:+.1f}%", va="center",
            fontsize=12, fontweight="bold",
            ha="left" if val>=0 else "right")

fig.suptitle(f"Ablation Study — Sentiment adds {avg_gain:+.1f}% accuracy on average",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(ABLATION_PNG, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close()
print(f"  Saved: {ABLATION_PNG}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save CSVs
# ══════════════════════════════════════════════════════════════════════════════
full_df.to_csv(CSV_PATH_FULL, index=False)
price_df.to_csv(CSV_PATH_PRCE, index=False)
ablation_df.to_csv(ABLATION_CSV, index=False)

print(f"  Saved: {CSV_PATH_FULL}")
print(f"  Saved: {CSV_PATH_PRCE}")
print(f"  Saved: {ABLATION_CSV}")
print(f"\n── FILES SAVED ──────────────────────────────────────────")
print(f"  models/ticker_results_v2.csv     ← full model results")
print(f"  models/ticker_results_v2.png     ← full model chart")
print(f"  models/ticker_results_price.csv  ← price only results")
print(f"  models/ablation_comparison.csv   ← side by side comparison")
print(f"  models/ablation_comparison.png   ← comparison chart")
print(f"\n  Use ticker_results_v2.csv and ablation_comparison.csv")
print(f"  for the Power BI dashboard.")
print("-" * 55)