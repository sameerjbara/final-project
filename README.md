# Stock Sentiment Prediction

> **Can today's news predict whether a stock price will go UP or DOWN tomorrow?**

A complete end-to-end machine learning pipeline that predicts next-day stock price direction using financial news sentiment. Built from raw API data collection through NLP scoring, feature engineering, modeling, and an interactive Power BI dashboard.

---

## Results

| Ticker | Accuracy | Sentiment Gain | Top Feature |
|--------|----------|---------------|-------------|
| NVDA | **68.0%** | +24.0% | article_count_norm |
| AMZN | **62.0%** | +18.0% | sentiment_roll3 |
| AAPL | **60.0%** | 0.0% | avg_finbert |
| GOOGL | 52.0% | +6.0% | std_finbert |
| META | 52.0% | -2.0% | std_finbert |
| NFLX | 50.0% | -4.0% | article_count_norm |
| **Average** | **57.3%** | **+7.0%** | |

**Price-only baseline:** 50.3% — essentially random.
**Sentiment adds:** +7.0% on average. For NVDA, sentiment alone adds +24 percentage points.

---

## Project Overview

### Tickers
`AAPL` `AMZN` `GOOGL` `META` `NFLX` `NVDA`

### Data
- **Period:** May 2024 → May 2026
- **Articles:** 33,338 news articles from 3 sources
- **Price data:** 3,126 daily price rows across 6 tickers

### Pipeline
```
Data Collection → Cleaning → Merging → FinBERT Scoring → Database
       → Feature Engineering → EDA → Modeling → Dashboard
```

---

## Repository Structure

```
stock-sentiment-predictor/
│
├── data/
│   ├── raw/                          # Raw API data
│   │   ├── stock_prices_raw.csv
│   │   ├── alphavantage_raw.csv
│   │   ├── polygon_raw.csv
│   │   └── guardian_raw.csv
│   │
│   └── processed/                    # Cleaned and engineered data
│       ├── stock_prices_clean.csv    # All 8 tickers (backup)
│       ├── stock_prices_6tickers.csv # Active 6 tickers
│       ├── news_master.csv           # 33,338 merged articles
│       ├── news_scored.csv           # Articles + FinBERT scores
│       ├── sentiment_daily.csv       # Daily sentiment (backup)
│       ├── sentiment_daily_6tickers.csv  # Active daily sentiment
│       └── features_v22.csv           # Final feature set (37 features)
│
├── models/
│   ├── ticker_results_v2.csv         # Full model results
│   ├── ticker_results_v2.png         # Results chart
│   ├── ticker_results_price.csv      # Price-only results
│   ├── ablation_comparison.csv       # Sentiment vs price comparison
│   └── ablation_comparison.png       # Ablation chart
│
├── src/
│   ├── collection/                   # Data collection scripts
│   │   ├── collect_prices.py
│   │   ├── alpha_vantage_collector.py
│   │   ├── polygon_collector.py
│   │   └── guardian_collector.py
│   │
│   ├── processing/                   # Data cleaning and processing
│   │   ├── clean_prices.py
│   │   ├── clean_alphavantage.py
│   │   ├── clean_polygon.py
│   │   ├── clean_guardian.py
│   │   ├── merge_news.py
│   │   ├── filter_tickers.py         # Removes JPM and MSFT
│   │   
│   │
│   ├── database/
│   │   └── setup_database.py         # Loads data into Neon PostgreSQL
│   │
│   ├── features/
│   │   └── feature_engineering_v2.py # V2 features (37) — ACTIVE
│   │
│   └── models/
│       ├── model_per_ticker.py       # Per-ticker ensemble — ACTIVE

│
├── notebooks/
│   ├── sentiment_scoring.py          # FinBERT scoring (run on Colab)
│   ├── eda_notebook.ipynb            # EDA on features V1 (Colab)
│
├── .env                              # API keys and database URL (not committed)
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Pipeline Details

### Phase 1 — Data Collection

Collected stock prices and news articles from multiple APIs.

```bash
python src/collection/collect_prices.py
python src/collection/alpha_vantage_collector.py
python src/collection/polygon_collector.py
python src/collection/guardian_collector.py
```

**Sources:**
- **Alpha Vantage** — 21,636 articles with pre-computed sentiment scores
- **Polygon** — 11,464 articles
- **The Guardian** — 519 articles
- **Yahoo Finance** — daily OHLCV price data

**Required API keys in `.env`:**
```
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key
GUARDIAN_API_KEY=your_key
```

---

### Phase 2 — Data Cleaning

Standardizes formats, removes duplicates, adds log returns.

```bash
python src/processing/clean_prices.py
python src/processing/clean_alphavantage.py
python src/processing/clean_polygon.py
python src/processing/clean_guardian.py
```

---

### Phase 3 — Merging

Combines all 3 news sources into one master file, removes cross-source duplicates.

```bash
python src/processing/merge_news.py
```

Output: `data/processed/news_master.csv` — 33,338 articles

---

### Phase 4 — FinBERT Scoring (Google Colab)

Scores all articles using `ProsusAI/finbert` on a T4 GPU. Takes approximately 11 minutes.

**Upload to Colab:**
- `notebooks/sentiment_scoring.py`
- `data/processed/news_master.csv`

**Run:**
```python
!python sentiment_scoring.py
```

**Download:**
- `news_scored.csv` — articles with FinBERT scores
- `sentiment_daily.csv` — one row per ticker per day

**FinBERT output per article:**
```
finbert_positive  — probability article is positive
finbert_negative  — probability article is negative
finbert_neutral   — probability article is neutral
finbert_score     — positive_prob - negative_prob (range: -1 to +1)
```

---

### Phase 4b — Ticker Filtering

Removes JPM and MSFT. Originals kept as backup.

```bash
python src/processing/filter_tickers.py
```

Output:
- `data/processed/stock_prices_6tickers.csv`
- `data/processed/sentiment_daily_6tickers.csv`

---

### Phase 5 — Database

Loads data into Neon PostgreSQL cloud database.

```bash
python src/database/setup_database.py
```

**Required in `.env`:**
```
DATABASE_URL=postgresql://user:password@host/dbname
```

**Tables created:**
- `stocks` — 6 rows
- `price_data` — 3,126 rows
- `sentiment_daily` — 2,899 rows

---

### Phase 6 — Feature Engineering V2

Builds 37 features combining price and sentiment signals.

```bash
python src/features/feature_engineering_v2.py
```

Output: `data/processed/features_v22.csv` — 4,168 rows × 40 columns

**Price features (10):**
`log_return`, `return_lag1`, `price_momentum_5d`, `volatility_5d`, `price_range`, `rsi_14`, `macd`, `volume_change`, `day_of_week`, `is_month_end`

**Sentiment features (21):**
`avg_finbert`, `std_finbert`, `bullish_ratio`, `bearish_ratio`, `avg_prescore`, `article_count`, `sentiment_lag1-5`, `sentiment_roll3/7`, `sentiment_momentum`, `sentiment_volume_interaction`, `high_news_day`, `sentiment_acceleration`, `article_count_norm`, `article_count_change`, `volatility_sentiment`, `prescore_finbert_diff`

**V2 improvements over V1:**
- Added: `sentiment_lag4`, `sentiment_lag5`, `sentiment_acceleration`, `article_count_norm`, `article_count_change`, `volatility_sentiment`, `prescore_finbert_diff`
- Removed: `prescore_finbert_agreement` (broken on neutral-fill days)

---

### Phase 7 — Modeling

Trains per-ticker ensemble. Runs full model AND price-only for comparison.

```bash
python src/models/model_per_ticker.py
```

**Model:** XGBoost (40%) + LightGBM (40%) + SVM (20%) ensemble per ticker

**Split:** 75% train / 15% validation / 10% test (date-based, no leakage)

**Output files:**
```
models/ticker_results_v2.csv     — full model results
models/ticker_results_price.csv  — price-only results
models/ablation_comparison.csv   — sentiment contribution per ticker
```

---

## Key Findings

### 1. Raw Sentiment is Weak But Real
- Raw directional accuracy: **53.1%** vs 50% random
- p-value = 0.71 — not statistically significant alone
- ML model amplifies this to **57.3%** average

### 2. Two Types of Stocks
- **Sentiment-driven:** NVDA, AMZN — news dominates prediction
- **Price-driven:** AAPL, NFLX — technical indicators dominate

### 3. Volume Noise
- NVDA gets 14.8 articles/day — articles conflict and cancel out
- Sweet spot: **4-7 articles/day** gives 55.7% raw accuracy
- More articles ≠ better signal

### 4. Source Agreement Matters
- When FinBERT and Alpha Vantage agree: **53.1%** accuracy
- When they disagree: **48.5%** — below random
- Disagreement is a reliable warning signal

### 5. Zero Article Days Hurt
- Days with no articles get neutral-filled with 0.0
- Accuracy on those days: **47.7%** — worse than random

---

## EDA Notebooks

### Raw EDA (`eda_raw_notebook.ipynb`)
- Uses only `stock_prices_6tickers.csv` and `sentiment_daily_6tickers.csv`
- No engineered features — pure raw data exploration
- 10 questions across 6 sections

**Upload to Colab:**
```
notebooks/eda_raw_notebook.ipynb
data/processed/stock_prices_6tickers.csv
data/processed/sentiment_daily_6tickers.csv
```

### Feature EDA (`eda_notebook.ipynb`)
- Uses `features_v22.csv`
- Explores engineered features and their relationship to the target
- Driven by modeling results

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-sentiment-predictor.git
cd stock-sentiment-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Add your API keys and database URL
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
transformers
torch
sqlalchemy
psycopg2-binary
python-dotenv
matplotlib
seaborn
scipy
requests
```

---

## How to Run the Full Pipeline

```bash
# 1. Collect data
python src/collection/collect_prices.py
python src/collection/alpha_vantage_collector.py
python src/collection/polygon_collector.py
python src/collection/guardian_collector.py

# 2. Clean data
python src/processing/clean_prices.py
python src/processing/clean_alphavantage.py
python src/processing/clean_polygon.py
python src/processing/clean_guardian.py

# 3. Merge news
python src/processing/merge_news.py

# 4. Score with FinBERT (run on Google Colab)
# Upload: news_master.csv + sentiment_scoring.py
# Download: news_scored.csv + sentiment_daily.csv

# 5. Filter tickers
python src/processing/filter_tickers.py

# 6. Load database
python src/database/setup_database.py

# 7. Engineer features
python src/features/feature_engineering_v2.py

# 8. Train and evaluate models
python src/models/model_per_ticker.py
```

---

## Design Decisions

| Decision | Reason |
|----------|--------|
| Per-ticker models | NVDA patterns ≠ META patterns — one global model adds noise |
| XGB + LGBM + SVM ensemble | No single model consistently dominates |
| ProsusAI/finbert | Industry standard for financial NLP |
| No article filtering | Every filtering approach hurt model accuracy |
| V2 features over V1 | EDA-driven improvements — +1% average accuracy |
| LSTM abandoned | <500 sequences per ticker — insufficient for deep learning |
| 75/15/10 split | Date-based split avoids data leakage from future data |
| Excluded JPM, MSFT | Macro-driven stocks — poor sentiment-to-price connection |

---

## Database Schema

```sql
CREATE TABLE stocks (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(100)
);

CREATE TABLE price_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume BIGINT,
    log_return FLOAT
);

CREATE TABLE sentiment_daily (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    trading_date DATE,
    avg_finbert FLOAT,
    std_finbert FLOAT,
    bullish_ratio FLOAT,
    bearish_ratio FLOAT,
    neutral_ratio FLOAT,
    avg_prescore FLOAT,
    article_count INTEGER
);
```

---

## Dashboard

Built in Power BI Desktop using the following data files:

```
models/ticker_results_v2.csv
models/ablation_comparison.csv
data/processed/features_v22.csv
data/processed/stock_prices_6tickers.csv
data/processed/sentiment_daily_6tickers.csv
```

**Pages:**
1. **Model Results** — accuracy per ticker, sentiment contribution, full vs price-only comparison
2. **Price & Coverage** — price history, article coverage, sentiment trend over time
3. **Summary** — KPI cards with headline numbers

---

## Future Work

- Add earnings dates as a feature — highest potential impact
- Expand to more tickers beyond 6 tech stocks
- Add Bloomberg, Reuters, CNBC as news sources
- Test FinBERT-tone and RoBERTa-financial for scoring
- Incorporate Twitter/Reddit sentiment
- Build real-time prediction API with daily automation
- Ticker-type classifier — sentiment-driven vs price-driven

---
## presentation link
https://drive.google.com/drive/folders/1S-YTV2dikhpvMsuft8Fw4xDSQfRWRrYF?usp=sharing

## License

MIT License — see `LICENSE` for details.
