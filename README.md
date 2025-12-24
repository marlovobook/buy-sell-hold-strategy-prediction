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



## Code Usage Example for rl v3

```python
import pandas as pd
from src.models.model_trainer_rl_v3 import ModelTrainerRL, TradingEnvRL
from src.utils.config_loader import load_config

# Load configuration
config = load_config("config/config.yaml")

# Load and prepare data
data = pd.read_csv('data/processed/YOUR_STOCK.csv')
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

# Split into train/test
split_idx = int(len(data) * 0.7)
df_train = data.iloc[:split_idx].copy()
df_test = data.iloc[split_idx:].copy()

# Initialize trainer
trainer = ModelTrainerRL(config['reinforcement_learning'])

# Prepare environments (handles feature scaling automatically)
env_train, env_test = trainer.prepare_environment(
    df_train, 
    df_test, 
    reward_func="sharpe"  # or "profit", "sortino", "cvar", "max_drawdown"
)

# Train PPO only
print("Training PPO...")
result = trainer.train_ppo(env_train)
model = result["model"]
print("Training Complete!")

# Evaluate with multiple seeds (recommended)
seeds = [42, 43, 44]
metrics_agg = trainer.evaluate_over_seeds(model, env_test, seeds, algorithm="PPO")

print("\n--- Multi-Seed Evaluation Results ---")
print(f"Total Return: {metrics_agg['total_return']['mean']:.2%} ± {metrics_agg['total_return']['ci95']}")
print(f"Sharpe Ratio: {metrics_agg['sharpe_ratio']['mean']:.3f} ± {metrics_agg['sharpe_ratio']['ci95']}")
print(f"Sortino Ratio: {metrics_agg['sortino_ratio']['mean']:.3f} ± {metrics_agg['sortino_ratio']['ci95']}")
print(f"Max Drawdown: {metrics_agg['max_drawdown']['mean']:.2%} ± {metrics_agg['max_drawdown']['ci95']}")

# Single run evaluation (if you only want one)
metrics_single = trainer.evaluate_model(model, env_test, algorithm="PPO", seed=42)
print("\n--- Single Run Evaluation ---")
print(f"Total Return: {metrics_single['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics_single['sharpe_ratio']:.3f}")

# Save the model and artifacts
trainer.save_models("models/ppo_artifacts")

# Use model for inference (generate actions on new data)
obs, _ = env_test.reset(seed=42)
done = False
actions = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions.append(float(action[0]))
    obs, _, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated

print(f"\nGenerated {len(actions)} trading actions")

```

## for inference later
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import joblib

# Load model and normalization
model = PPO.load("models/ppo_artifacts/ppo_model")
vec_norm = VecNormalize.load("models/ppo_artifacts/ppo_vecnormalize.pkl")
scaler = joblib.load("models/ppo_artifacts/feature_scaler.joblib")

# Use on new data
# 1. Scale new features with loaded scaler
# 2. Reset env with VecNormalize wrapper
# 3. Call model.predict()
```


## Usage Example for train_all
```python
# In your notebook or script:
trainer = ModelTrainerRL(config['reinforcement_learning'])

# Standard train/test with multi-seed evaluation
results = trainer.train_all(df, test_size=0.2, reward_func="sharpe")

# Or walk-forward evaluation
wf_results = trainer.walk_forward_evaluation(
    df,
    window_size=200,
    step_size=50,
    reward_func="sharpe",
    algorithm="sac",
    seeds=[42, 43, 44],
)

# Save everything (models + stats + scaler)
trainer.save_models("models/rl_artifacts")

```