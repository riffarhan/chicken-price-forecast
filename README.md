# Chicken Price Forecast

An end-to-end machine learning system that predicts weekly chicken prices across 15 Indonesian provinces using XGBoost, with SHAP-based explainability and a Streamlit dashboard.

## Features

- **XGBoost regression** with 23 engineered features (lags, rolling stats, seasonality, interactions)
- **SHAP explanations** — understand *why* each prediction was made
- **Interactive Streamlit dashboard** with province-level charts, alerts, and CSV downloads
- **Automated weekly pipeline** with performance monitoring and degradation alerts
- **Synthetic data generator** that produces realistic price patterns with holiday effects, supply shocks, and regional variation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/riffarhan/chicken-price-forecast.git
cd chicken-price-forecast

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# 1. Generate sample dataset
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
├── generate_dataset.py     # Create synthetic training data
├── train_model.py          # Train XGBoost model with baseline comparison
├── predict.py              # Production prediction pipeline
├── dashboard.py            # Streamlit web dashboard
├── requirements.txt        # Python dependencies
├── scripts/
│   ├── run_prediction.sh   # Cron automation (Mac/Linux)
│   └── run_prediction.bat  # Task Scheduler automation (Windows)
├── data/                   # Generated datasets (gitignored)
├── models/                 # Trained model artifacts (gitignored)
├── outputs/                # Predictions, metrics, SHAP summaries
└── logs/                   # Execution logs
```

## How It Works

### Data Generation

`generate_dataset.py` creates a realistic synthetic dataset with:

| Factor | Description |
|--------|-------------|
| **Seasonality** | Q4 price spikes, Q1 normalization |
| **Holidays** | Idul Fitri, Christmas/New Year, Independence Day |
| **Feed costs** | Correlated with chicken prices, trending upward |
| **Supply shocks** | Random 5% weekly probability |
| **Weather** | Three severity levels affecting prices |
| **Regional variation** | Different base prices and logistics costs per province |

Output: `data/chicken_price_dataset.csv` — 1,560 rows (104 weeks × 15 provinces)

### Model Training

`train_model.py` trains an XGBoost regressor using 23 features:

- **Lagged prices** (1, 2, 4 weeks) and **lagged feed costs**
- **Rolling statistics** (4-week mean and std)
- **Temporal encodings** (week, month, quarter, cyclical sin/cos)
- **Interaction features** (feed × holiday, supply × weather)
- **Province encoding** and **logistics cost**

The model is benchmarked against a linear regression baseline. Typical results on the synthetic dataset:

| Metric | Linear Regression | XGBoost |
|--------|-------------------|---------|
| RMSE | ~1,400 IDR/kg | ~867 IDR/kg |
| MAPE | ~5% | ~3.2% |
| R² | ~0.92 | ~0.97 |

### Prediction Pipeline

`predict.py` runs the full inference loop:

1. Load latest data and apply feature engineering
2. Generate predictions with the trained model
3. Calculate SHAP values for the most recent observation per province
4. Validate performance against the RMSE threshold (1,200 IDR/kg)
5. Save predictions, SHAP summaries, and append metrics history

### Dashboard

`dashboard.py` provides:

- Real-time performance metrics (RMSE, MAPE, status)
- Province-level actual vs. predicted price charts
- SHAP-based explanation of prediction drivers
- Cross-province price comparison
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

1. Open Task Scheduler → Create Basic Task
2. Trigger: Weekly, Monday, 6:00 AM
3. Action: Start a program → browse to `scripts\run_prediction.bat`

## When to Retrain

The pipeline logs a `DEGRADED` status when RMSE exceeds the threshold. Retrain if:

- RMSE > 1,200 IDR/kg for 2+ consecutive weeks
- 4 weeks of upward RMSE trend
- Quarterly scheduled retraining

```bash
python train_model.py
```

## Adapting to Real Data

This project ships with a synthetic data generator for demonstration. To use real price data:

1. Replace `data/chicken_price_dataset.csv` with your own CSV using the same column schema
2. Re-run `python train_model.py` to retrain on your data
3. Run `python predict.py` to generate forecasts

Required columns: `date`, `province`, `chicken_price`, `feed_price`, `supply_index`, `weather_severity`, `is_holiday`, `week_of_year`, `month`, `quarter`, `year`, `logistics_cost`

## Tech Stack

- **ML**: XGBoost, scikit-learn, SHAP
- **Data**: pandas, NumPy, SciPy
- **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
- **Python**: 3.10+

## License

[MIT](LICENSE)
