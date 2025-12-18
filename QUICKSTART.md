# Quick Start Guide

This guide will help you get started with the Buy/Sell/Hold Strategy Prediction system.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marlovobook/buy-sell-hold-strategy-prediction.git
cd buy-sell-hold-strategy-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to customize:

- **Stock symbols**: Add or remove symbols to analyze
- **Date range**: Set start and end dates for historical data
- **Model parameters**: Adjust hyperparameters for each model
- **Trading strategy**: Set thresholds and initial capital

## Usage

### Option 1: Run Complete Pipeline

Execute the full ML pipeline (recommended for first-time users):

```bash
python pipeline.py
```

This will:
1. Download historical stock data from Yahoo Finance
2. Calculate technical indicators
3. Train multiple ML models (Random Forest, XGBoost, LSTM)
4. Select the best performing model
5. Run portfolio backtesting
6. Save all results to `data/` and `models/` directories

### Option 2: Quick Demo Setup

For a quick demo with pre-generated data:

```bash
python setup_demo.py
```

This creates sample data for testing the Streamlit dashboard without internet access.

### Option 3: Run Tests

To verify the installation:

```bash
python test_pipeline.py
```

This runs a quick test with mock data to ensure everything is working correctly.

## Visualize Results

After running the pipeline or demo setup, launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Features

The Streamlit dashboard has 4 main tabs:

### 1. Data Overview
- Historical price charts
- Summary statistics
- Recent data preview

### 2. Model Performance
- Comparison of all trained models
- Accuracy metrics
- Classification reports
- Confusion matrices

### 3. Portfolio Analysis
- Strategy performance vs buy-and-hold
- Portfolio value over time
- Performance metrics (Sharpe ratio, drawdown, etc.)
- Trade statistics

### 4. Technical Analysis
- Technical indicator charts (SMA, RSI, MACD, Bollinger Bands)
- Current indicator values
- Signal interpretation

## Project Structure

```
buy-sell-hold-strategy-prediction/
├── config/              # Configuration files
├── data/               # Data storage
│   ├── raw/           # Raw data from yfinance
│   └── processed/     # Processed data with features
├── models/            # Trained models
├── src/               # Source code
│   ├── data/         # Data collection modules
│   ├── models/       # ML training and backtesting
│   └── utils/        # Utility functions
├── pipeline.py       # Main ML pipeline
├── app.py           # Streamlit dashboard
├── test_pipeline.py # Test script
└── setup_demo.py    # Demo data generator
```

## Troubleshooting

### Internet Connection Issues

If you can't access Yahoo Finance:
```bash
python test_pipeline.py  # Uses mock data
```

### Model Training Takes Too Long

Edit `config/config.yaml`:
- Reduce `n_estimators` for Random Forest and XGBoost
- Reduce `epochs` for LSTM
- Use shorter date range

### Streamlit App Not Loading

1. Check that data files exist in `data/processed/`
2. Run `python setup_demo.py` to generate sample data
3. Check for error messages in the terminal

## Next Steps

1. **Customize the strategy**: Modify the labeling logic in `src/data/data_collector.py`
2. **Add more models**: Extend `src/models/model_trainer.py` with additional algorithms
3. **Improve features**: Add more technical indicators in the data collector
4. **Optimize hyperparameters**: Use GridSearch or RandomSearch for tuning
5. **Paper trading**: Integrate with a broker API for live testing

## Support

For issues and questions:
- Check the main [README.md](README.md)
- Review the code documentation in each module
- Open an issue on GitHub

## Disclaimer

This software is for educational purposes only. Not financial advice.
Always do your own research before making investment decisions.
