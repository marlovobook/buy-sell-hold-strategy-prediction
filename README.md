# Buy/Sell/Hold Strategy Prediction

A comprehensive Machine Learning pipeline for predicting buy/sell/hold trading signals using software factory methodology. This project integrates data collection from Yahoo Finance (yfinance), multiple ML models for prediction, and portfolio backtesting with vectorbt, all visualized through an interactive Streamlit dashboard.

## 🎯 Features

- **Data Collection**: Automated historical stock data collection using yfinance
- **Technical Indicators**: Calculates SMA, RSI, MACD, Bollinger Bands, and more
- **Multiple ML Models**: 
  - Random Forest Classifier
  - XGBoost Classifier
  - LSTM Neural Network
- **Model Selection**: Automatic selection of best performing model
- **Portfolio Backtesting**: Strategy evaluation using vectorbt
- **Interactive Dashboard**: Real-time visualization with Streamlit
- **Performance Metrics**: Comprehensive analysis including Sharpe ratio, drawdown, win rate

## 📁 Project Structure

```
buy-sell-hold-strategy-prediction/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw data from yfinance
│   └── processed/               # Processed data with features
├── models/                      # Trained models
├── src/
│   ├── data/
│   │   └── data_collector.py   # Data collection and preprocessing
│   ├── models/
│   │   ├── model_trainer.py    # Model training and selection
│   │   └── backtester.py       # Portfolio backtesting with vectorbt
│   └── utils/
│       └── config_loader.py    # Configuration utilities
├── pipeline.py                  # Main ML pipeline
├── app.py                       # Streamlit dashboard
└── requirements.txt             # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/marlovobook/buy-sell-hold-strategy-prediction.git
cd buy-sell-hold-strategy-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Edit `config/config.yaml` to customize:
- Stock symbols to analyze
- Date ranges
- Model parameters
- Trading strategy settings

### Usage

#### 1. Run the ML Pipeline

Execute the complete pipeline (data collection, training, backtesting):

```bash
python pipeline.py
```

This will:
- Download historical stock data
- Calculate technical indicators
- Train multiple ML models
- Select the best model
- Run backtesting
- Save results to `models/` and `data/` directories

#### 2. Launch the Streamlit Dashboard

Visualize results in an interactive dashboard:

```bash
streamlit run app.py
```

The dashboard includes:
- **Data Overview**: Historical prices and statistics
- **Model Performance**: Comparison of different models
- **Portfolio Analysis**: Backtesting results with vectorbt
- **Technical Analysis**: Indicators and signals

## 📊 Software Factory Methodology

This project follows software factory principles:

1. **Modular Design**: Separated concerns (data, models, utils)
2. **Configuration Management**: Centralized config file
3. **Automated Pipeline**: End-to-end automation from data to results
4. **Quality Assurance**: Model evaluation and comparison
5. **Visualization**: Interactive dashboard for stakeholders
6. **Reproducibility**: Saved models and configurations

## 🔧 Components

### Data Collection (`src/data/data_collector.py`)
- Downloads data from Yahoo Finance using yfinance
- Calculates technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Creates buy/sell/hold labels based on future returns
- Handles data preprocessing and validation

### Model Training (`src/models/model_trainer.py`)
- Trains multiple models: Random Forest, XGBoost, LSTM
- Performs train-test split with stratification
- Evaluates models using accuracy and classification metrics
- Automatically selects best performing model
- Saves models and results

### Backtesting (`src/models/backtester.py`)
- Converts predictions to trading signals
- Simulates portfolio performance using vectorbt
- Calculates comprehensive metrics (returns, Sharpe ratio, drawdown)
- Compares strategy with buy-and-hold benchmark

### Streamlit Dashboard (`app.py`)
- Interactive visualization of data and results
- Real-time model performance comparison
- Portfolio value charts
- Technical indicator plots

## 📈 Example Results

After running the pipeline, you'll see:
- Model accuracy comparisons
- Portfolio returns vs buy-and-hold
- Sharpe ratio and risk metrics
- Win rate and trade statistics

## 🛠️ Technologies Used

- **yfinance**: Stock data collection
- **scikit-learn**: Traditional ML models
- **XGBoost**: Gradient boosting
- **TensorFlow/Keras**: Deep learning (LSTM)
- **vectorbt**: Portfolio backtesting
- **Streamlit**: Web dashboard
- **Plotly**: Interactive visualizations
- **pandas/numpy**: Data manipulation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always do your own research and consult with financial professionals before making investment decisions.