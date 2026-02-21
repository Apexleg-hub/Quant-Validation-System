import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Import custom modules
from data_loader import load_data, add_features
from models.logistic import run_logistic
from models.svm import run_svm
from backtest.engine import run_backtest

st.set_page_config(page_title="Trading Dashboard", layout="wide")

# Sidebar: data loading controls
with st.sidebar:
    st.header("Data Settings")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    end_date = datetime.today()
    start_date = st.date_input("Start Date", value=end_date - timedelta(days=365))
    
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            df_raw = load_data(ticker, start_date, end_date)
            df_feat = add_features(df_raw)
            st.session_state['df_raw'] = df_raw
            st.session_state['df_feat'] = df_feat
            st.success("Data loaded!")

# Check if data exists in session state
if 'df_feat' not in st.session_state:
    st.warning("Please load data from the sidebar.")
    st.stop()

df_raw = st.session_state['df_raw']
df_feat = st.session_state['df_feat']

# Prepare features and target for modeling
features = ['return_lag1', 'return_lag2', 'return_lag3', 'SMA_10', 'SMA_30', 
            'EMA_10', 'RSI', 'MACD', 'MACD_signal', 'ATR']
X = df_feat[features]
y = df_feat['target']

# Train/test split (use last 20% for testing)
split_idx = int(len(df_feat) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Create tabs for each model
tabs = st.tabs(["📈 Overview", "🤖 Logistic Regression", "🤖 SVM", "🔁 Backtest (Logistic)", "🔁 Backtest (SVM)"])

with tabs[0]:
    # Overview: candlestick chart, metrics, etc. (same as before)
    st.subheader("Price Chart")
    fig = go.Figure(data=[go.Candlestick(x=df_raw['Date'],
                    open=df_raw['Open'], high=df_raw['High'],
                    low=df_raw['Low'], close=df_raw['Close'])])
    st.plotly_chart(fig, use_container_width=True)

# Logistic Regression Tab
with tabs[1]:
    st.subheader("Logistic Regression Results")
    
    # Cache model training so it only runs once
    @st.cache_resource
    def train_logistic():
        return run_logistic(X_train, X_test, y_train, y_test, features)
    
    logistic_results = train_logistic()
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{logistic_results['accuracy']:.2%}")
    
    # Show feature importance (coefficients)
    st.subheader("Feature Coefficients")
    fig = px.bar(logistic_results['importance'], x='coefficient', y='feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    # Optionally show classification report as dataframe
    st.subheader("Classification Report")
    report_df = pd.DataFrame(logistic_results['report']).transpose()
    st.dataframe(report_df)

# SVM Tab
with tabs[2]:
    st.subheader("SVM Results")
    
    @st.cache_resource
    def train_svm():
        return run_svm(X_train, X_test, y_train, y_test, features)
    
    svm_results = train_svm()
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{svm_results['accuracy']:.2%}")
    
    # SVM might not have feature importance directly; you could use permutation importance or SHAP.
    # For simplicity, we'll just show the confusion matrix or classification report.
    st.subheader("Classification Report")
    report_df = pd.DataFrame(svm_results['report']).transpose()
    st.dataframe(report_df)

# Backtest for Logistic Regression
with tabs[3]:
    st.subheader("Backtest: Logistic Regression Strategy")
    
    # Get predictions on the entire dataset (or test set) for backtest
    # Here we use the model to predict on the whole feature set
    logistic_model = logistic_results['model']
    full_predictions = logistic_model.predict(X)
    
    # Run backtest
    backtest_log = run_backtest(
        predictions=full_predictions,
        prices=df_feat['Close'].values,
        dates=df_feat['Date'].values
    )
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy Return", f"{backtest_log['total_strategy_return']:.2%}")
    col2.metric("Buy & Hold Return", f"{backtest_log['total_asset_return']:.2%}")
    col3.metric("Sharpe Ratio", f"{backtest_log['sharpe']:.2f}")
    col4.metric("Max Drawdown", f"{backtest_log['max_drawdown']:.2%}")
    
    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_log['equity_curve']['date'], 
                             y=backtest_log['equity_curve']['cum_asset'],
                             name="Buy & Hold"))
    fig.add_trace(go.Scatter(x=backtest_log['equity_curve']['date'], 
                             y=backtest_log['equity_curve']['cum_strategy'],
                             name="Strategy"))
    fig.update_layout(title="Equity Curve")
    st.plotly_chart(fig, use_container_width=True)

# Backtest for SVM (similar)
with tabs[4]:
    st.subheader("Backtest: SVM Strategy")
    
    svm_model = svm_results['model']
    full_predictions_svm = svm_model.predict(X)
    
    backtest_svm = run_backtest(
        predictions=full_predictions_svm,
        prices=df_feat['Close'].values,
        dates=df_feat['Date'].values
    )
    
    # Display metrics and equity curve...
    # (same structure as above)