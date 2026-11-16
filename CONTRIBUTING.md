# Contributing

## Setup

```bash
git clone https://github.com/riffarhan/chicken-price-forecast.git
cd chicken-price-forecast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make your change and add or adjust tests in `tests/`
3. Run `pytest` and `ruff check src tests app`
4. If you change features or the model, rerun `chicken-forecast train` and
   `chicken-forecast ablation`, and report the backtest change in your PR
5. Open a pull request

## Ground rules for new features

This project only uses features that are:

- **Backed by literature.** Add the paper to `docs/literature.md` with a full
  citation, and only cite what you have actually read.
- **Built from public, downloadable data.** Add a downloader under
  `src/chicken_forecast/data/`. No hand-typed or synthetic data.
- **Free of look-ahead.** Feature values at week *t* may only use data
  published by week *t*. `tests/test_features.py` has a leakage test; extend it.
- **Useful out of sample.** The feature group must improve the rolling-origin
  backtest in `chicken-forecast ablation`.

## Ideas

- DOC (day-old chick) and live-bird prices, if a public source appears
  (significant in Fitrimawati et al., 2024)
- Domestic maize prices from PIHPS-like sources instead of global benchmarks
- Full MinT covariance and probabilistic reconciliation (the current
  reconciliation uses a diagonal variance proxy)
- Pre-registered test on post-Nov-2026 data: quantile median as the point
  forecast, and dropping the heat-stress features
