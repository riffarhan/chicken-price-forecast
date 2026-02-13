# Contributing

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/riffarhan/chicken-price-forecast.git
cd chicken-price-forecast
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test locally: run the full pipeline (`generate_dataset.py` → `train_model.py` → `predict.py` → `streamlit run dashboard.py`)
5. Commit with a clear message
6. Open a pull request

## Ideas for Contributions

- Add new features to the model (e.g., inflation data, fuel prices)
- Improve the dashboard with additional visualizations
- Add unit tests
- Support additional ML algorithms (LightGBM, CatBoost)
- Add Docker support
- Connect to a real data source

## Code Style

- Follow PEP 8
- Use type hints where practical
- Keep functions focused and well-documented
