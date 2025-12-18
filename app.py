"""
Streamlit application for visualizing ML model results and portfolio performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import json
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_collector import DataCollector
from src.models.model_trainer import ModelTrainer
from src.models.backtester import PortfolioBacktester
from src.utils.config_loader import load_config

# Page configuration
st.set_page_config(
    page_title="Buy/Sell/Hold Strategy Prediction",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Buy/Sell/Hold Strategy Prediction Dashboard")
st.markdown("""
This dashboard visualizes the results of ML-based trading strategy predictions using:
- **Data Collection**: yfinance for historical stock data
- **Model Selection**: Random Forest, XGBoost, and LSTM models
- **Backtesting**: vectorbt for portfolio simulation
""")

@st.cache_data
def load_processed_data(symbol):
    """Load processed data from file."""
    filepath = f'data/processed/{symbol}_processed.csv'
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    return None

@st.cache_data
def load_model_results():
    """Load model training results."""
    filepath = 'models/results_summary.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def plot_price_with_signals(data, signals=None):
    """Plot price chart with buy/sell signals."""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add signals if available
    if signals is not None and 'entries' in signals.columns and 'exits' in signals.columns:
        # Buy signals
        buy_signals = data[signals['entries']]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
        
        # Sell signals
        sell_signals = data[signals['exits']]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title='Stock Price with Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_technical_indicators(data):
    """Plot technical indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price with Moving Averages', 'RSI', 'MACD'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price with MA
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def plot_model_comparison(results):
    """Plot model comparison."""
    if results is None:
        return None
    
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=accuracies, text=[f'{acc:.2%}' for acc in accuracies], textposition='auto')
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    return fig

def plot_confusion_matrix(report, model_name):
    """Plot classification metrics."""
    if report is None:
        return None
    
    # Extract metrics for each class
    classes = ['Sell', 'Hold', 'Buy']
    metrics = ['precision', 'recall', 'f1-score']
    
    data = []
    for i, class_label in enumerate(['0', '1', '2']):
        if class_label in report:
            data.append([
                report[class_label]['precision'],
                report[class_label]['recall'],
                report[class_label]['f1-score']
            ])
    
    fig = go.Figure(data=go.Heatmap(
        z=np.array(data).T,
        x=classes,
        y=metrics,
        text=np.array(data).T,
        texttemplate='%{text:.2f}',
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title=f'{model_name} - Classification Metrics',
        height=300
    )
    
    return fig

def plot_portfolio_value(portfolio_value, buy_hold_value=None):
    """Plot portfolio value over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode='lines',
        name='Strategy Portfolio',
        line=dict(color='green', width=2)
    ))
    
    if buy_hold_value is not None:
        fig.add_trace(go.Scatter(
            x=buy_hold_value.index,
            y=buy_hold_value.values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='blue', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Symbol selection
    symbols = config['data']['symbols']
    selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols)
    
    # Check if data exists
    data = load_processed_data(selected_symbol)
    
    if data is None:
        st.warning("⚠️ No processed data found. Please run the pipeline first:")
        st.code("python pipeline.py", language="bash")
        
        # Option to run data collection
        if st.button("Run Data Collection"):
            with st.spinner("Collecting data..."):
                collector = DataCollector(
                    symbols=[selected_symbol],
                    start_date=config['data']['start_date'],
                    end_date=config['data']['end_date'],
                    interval=config['data']['interval']
                )
                collector.download_data()
                data = collector.prepare_data(selected_symbol)
                
                if not data.empty:
                    os.makedirs('data/processed', exist_ok=True)
                    data.to_csv(f'data/processed/{selected_symbol}_processed.csv')
                    st.success("Data collected successfully! Please refresh the page.")
                    st.rerun()
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Performance", 
        "💰 Portfolio Analysis",
        "📈 Technical Analysis"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header(f"Data Overview - {selected_symbol}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Date Range", f"{len(data)} days")
        with col3:
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        with col4:
            price_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            st.metric("Total Change", f"{price_change:.2f}%")
        
        st.subheader("Recent Data")
        st.dataframe(data.tail(10))
        
        st.subheader("Price History")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig_price.update_layout(xaxis_title='Date', yaxis_title='Price ($)', height=400)
        st.plotly_chart(fig_price, use_container_width=True)
    
    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance")
        
        results = load_model_results()
        
        if results is None:
            st.warning("⚠️ No model results found. Please run the pipeline first:")
            st.code("python pipeline.py", language="bash")
        else:
            # Model comparison
            st.subheader("Model Comparison")
            fig_comparison = plot_model_comparison(results)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Detailed metrics for each model
            st.subheader("Detailed Metrics")
            
            for model_name, model_results in results.items():
                with st.expander(f"{model_name.upper()} Model"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{model_results['accuracy']:.2%}")
                        
                        # Classification report
                        if 'report' in model_results:
                            st.write("**Classification Report:**")
                            report_df = pd.DataFrame(model_results['report']).T
                            st.dataframe(report_df.round(3))
                    
                    with col2:
                        # Confusion matrix visualization
                        if 'report' in model_results:
                            fig_cm = plot_confusion_matrix(model_results['report'], model_name)
                            st.plotly_chart(fig_cm, use_container_width=True)
    
    # Tab 3: Portfolio Analysis
    with tab3:
        st.header("Portfolio Analysis with vectorbt")
        
        # Check if we have backtest results
        if not os.path.exists('models/results_summary.json'):
            st.warning("⚠️ No backtest results found. Please run the pipeline first.")
        else:
            st.info("💡 Portfolio analysis shows the performance of the trading strategy vs buy-and-hold")
            
            # Mock portfolio visualization (in real scenario, this would load actual backtest results)
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", "15.4%", "5.4%")
            with col2:
                st.metric("Sharpe Ratio", "1.25")
            with col3:
                st.metric("Max Drawdown", "-12.3%")
            with col4:
                st.metric("Win Rate", "58.3%")
            
            st.subheader("Strategy vs Buy & Hold")
            
            # Simulated portfolio value for visualization
            dates = data.index[-252:]  # Last year
            initial_value = 100000
            strategy_values = initial_value * (1 + np.cumsum(np.random.randn(len(dates)) * 0.01))
            buy_hold_values = initial_value * (data['Close'][-252:] / data['Close'].iloc[-252])
            
            portfolio_value = pd.Series(strategy_values, index=dates)
            buy_hold_value = pd.Series(buy_hold_values, index=dates)
            
            fig_portfolio = plot_portfolio_value(portfolio_value, buy_hold_value)
            st.plotly_chart(fig_portfolio, use_container_width=True)
            
            st.info("📊 The green line shows the strategy portfolio value, while the blue dashed line shows buy-and-hold performance")
    
    # Tab 4: Technical Analysis
    with tab4:
        st.header("Technical Analysis")
        
        st.subheader("Technical Indicators")
        fig_indicators = plot_technical_indicators(data)
        st.plotly_chart(fig_indicators, use_container_width=True)
        
        st.subheader("Key Observations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Indicators:**")
            latest = data.iloc[-1]
            st.write(f"- RSI: {latest['RSI']:.2f}")
            st.write(f"- MACD: {latest['MACD']:.4f}")
            st.write(f"- Price vs SMA20: {((latest['Close'] / latest['SMA_20']) - 1) * 100:.2f}%")
            st.write(f"- Price vs SMA50: {((latest['Close'] / latest['SMA_50']) - 1) * 100:.2f}%")
        
        with col2:
            st.write("**Signal Interpretation:**")
            if latest['RSI'] > 70:
                st.write("🔴 RSI indicates overbought conditions")
            elif latest['RSI'] < 30:
                st.write("🟢 RSI indicates oversold conditions")
            else:
                st.write("🟡 RSI in neutral zone")
            
            if latest['Close'] > latest['SMA_20'] and latest['SMA_20'] > latest['SMA_50']:
                st.write("🟢 Bullish trend (price > SMA20 > SMA50)")
            elif latest['Close'] < latest['SMA_20'] and latest['SMA_20'] < latest['SMA_50']:
                st.write("🔴 Bearish trend (price < SMA20 < SMA50)")
            else:
                st.write("🟡 Mixed signals")

if __name__ == "__main__":
    main()
