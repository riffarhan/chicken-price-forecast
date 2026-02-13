# Chicken Price Forecast

An end-to-end machine learning system that predicts weekly chicken prices across 15 Indonesian provinces using XGBoost, with SHAP-based explainability and a Streamlit dashboard.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Features

- **XGBoost regression** with 23 engineered features (lags, rolling stats, seasonality, interactions)
- **SHAP explanations** — understand *why* each prediction was made
- **Interactive Streamlit dashboard** with province-level charts, alerts, and CSV downloads
- **Automated weekly pipeline** with performance monitoring and degradation alerts
- **Shared feature engineering** — single source of truth across training and inference
- **Deterministic province encoding** — consistent predictions regardless of input subset

## Quick Start

```bash
git clone https://github.com/riffarhan/chicken-price-forecast.git
cd chicken-price-forecast

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 1. Generate sample dataset (1,560 rows — 104 weeks x 15 provinces)
python generate_dataset.py

# 2. Train the model
python train_model.py

# 3. Run the prediction pipeline
python predict.py

# 4. Launch the dashboard
streamlit run dashboard.py
```

The dashboard opens at **http://localhost:8501**.

## Project Structure

```
chicken-price-forecast/
├── config.py               # Shared paths, constants, hyperparameters
├── features.py             # Shared feature engineering pipeline
├── generate_dataset.py     # Create synthetic training data
├── train_model.py          # Train XGBoost with baseline comparison
├── predict.py              # Production prediction + SHAP + monitoring
├── dashboard.py            # Streamlit web dashboard
├── requirements.txt
├── scripts/
│   ├── run_prediction.sh   # Cron automation (Mac/Linux)
│   └── run_prediction.bat  # Task Scheduler automation (Windows)
├── data/                   # Generated datasets
├── models/                 # Trained model artifacts
├── outputs/                # Predictions, metrics, SHAP summaries
└── logs/                   # Execution logs
```

## How It Works

### Architecture

```
generate_dataset.py ──▸ data/chicken_price_dataset.csv
                              │
                              ▼
train_model.py ────────▸ models/chicken_price_xgboost_model.json
                              │
                              ▼
predict.py ────────────▸ outputs/predictions_latest.csv
                         outputs/model_metrics.csv
                         outputs/shap_summary_*.csv
                              │
                              ▼
dashboard.py ──────────▸ http://localhost:8501
```

All scripts share feature engineering via `features.py` and configuration via `config.py`, so the transformation logic is defined once and stays consistent.

### Data Generation

`generate_dataset.py` creates a realistic synthetic dataset with:

| Factor | Description |
|--------|-------------|
| Seasonality | Q4 price spikes, Q1 normalization |
| Holidays | Idul Fitri, Christmas/New Year, Independence Day |
| Feed costs | Correlated with chicken prices, trending upward |
| Supply shocks | Random 5% weekly probability |
| Weather | Three severity levels affecting prices |
| Regional variation | Different base prices and logistics costs per province |

### Feature Engineering

The shared `features.py` module creates 23 features from the raw data:

- **Lagged prices** (1, 2, 4 weeks) and **lagged feed costs** (1, 2 weeks)
- **Rolling statistics** — 4-week mean and standard deviation
- **Change features** — week-over-week price and feed changes
- **Ratio** — feed-to-price ratio
- **Province encoding** — deterministic integer mapping (not LabelEncoder)
- **Interaction features** — feed x holiday, supply x weather
- **Cyclical encodings** — sin/cos of week-of-year

### Model Performance

Typical results on the synthetic dataset:

| Metric | Linear Regression | XGBoost |
|--------|-------------------|---------|
| RMSE | ~1,400 IDR/kg | ~867 IDR/kg |
| MAPE | ~5% | ~3.2% |
| R² | ~0.92 | ~0.97 |

### Dashboard

The Streamlit dashboard provides:

- Real-time performance metrics (RMSE, MAPE, health status)
- Province-level actual vs. predicted price charts
- SHAP-based explanation of prediction drivers per province
- Cross-province price comparison bar chart
- Performance trend monitoring with alert threshold
- CSV download for predictions and SHAP analysis

## Scheduling

### Mac / Linux (Cron)

```bash
chmod +x scripts/run_prediction.sh

# Run every Monday at 6 AM
crontab -e
0 6 * * 1 /path/to/chicken-price-forecast/scripts/run_prediction.sh >> /path/to/chicken-price-forecast/logs/cron.log 2>&1
```

### Windows (Task Scheduler)

1. Open Task Scheduler and create a basic task
2. Trigger: Weekly, Monday, 6:00 AM
3. Action: Start a program — browse to `scripts\run_prediction.bat`

## Retraining

The pipeline logs a `DEGRADED` status when RMSE exceeds the threshold (1,200 IDR/kg). Retrain when:

- RMSE > 1,200 for 2+ consecutive weeks
- 4 weeks of upward RMSE trend
- Quarterly scheduled retraining

```bash
python train_model.py
```

## Adapting to Real Data

This project ships with a synthetic data generator for demonstration. To use your own data:

1. Prepare a CSV with these columns: `date`, `province`, `chicken_price`, `feed_price`, `supply_index`, `weather_severity`, `is_holiday`, `week_of_year`, `month`, `quarter`, `year`, `logistics_cost`
2. Save it to `data/chicken_price_dataset.csv`
3. Run `python train_model.py` to train on your data
4. Run `python predict.py` to generate forecasts

The feature engineering pipeline (`features.py`) will create all derived columns automatically. Province names must match those defined in `config.py` — add new provinces there if needed.

## Tech Stack

| Category | Libraries |
|----------|-----------|
| ML | XGBoost, scikit-learn, SHAP |
| Data | pandas, NumPy, SciPy |
| Dashboard | Streamlit, Plotly |
| Visualization | Matplotlib, Seaborn |

Requires **Python 3.10+**.

## License

[MIT](LICENSE)
