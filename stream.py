

"""
Quant Validation System — Trading Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ── Custom modules ────────────────────────────────────────────────────────────
from ingestion.data_cleaner import load_and_clean_data, clean_data, create_features
from ml_models.logisticRegression import run_logistic
from ml_models.svm import run_svm
from ml_models.linear_regression import run_linear_regression
from ml_models.RandomForest import run_random_forest
from ml_models.preditvolitilty import predict_volatility
from ml_models.atr_risk import apply_atr_risk
from backtesting.backtest_example import run_backtest

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Validation System",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark terminal feel */
    [data-testid="stAppViewContainer"] { background: #0d0f14; }
    [data-testid="stSidebar"]          { background: #111318; border-right: 1px solid #1e2130; }

    /* Typography */
    html, body, [class*="css"]  { font-family: 'IBM Plex Mono', monospace; color: #c9d1d9; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricValue"]  { color: #58a6ff; font-size: 1.6rem; }
    [data-testid="stMetricLabel"]  { color: #8b949e; font-size: 0.75rem; letter-spacing: .08em; }

    /* Tabs */
    [data-testid="stTabs"] button {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #8b949e;
        border-bottom: 2px solid transparent;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }

    /* Buttons */
    [data-testid="baseButton-secondary"] {
        background: #21262d;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
    }
    [data-testid="baseButton-secondary"]:hover { border-color: #58a6ff; }

    /* Section headers */
    h2, h3 { color: #e6edf3; letter-spacing: -0.02em; }
    .section-label {
    font-size: 0.7rem;
        letter-spacing: .12em;
        color: #58a6ff;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .divider { border-top: 1px solid #21262d; margin: 1.5rem 0; }
</style>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    'return_lag1', 'return_lag2', 'return_lag3',
    'SMA_10', 'SMA_30', 'EMA_10',
    'RSI', 'MACD', 'MACD_signal', 'ATR',
]
TRAIN_RATIO = 0.8

# ── Plotly dark theme helper ───────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#0d0f14",
    font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
    xaxis=dict(gridcolor="#1e2130", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2130", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)

def styled_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT)
    return fig

# ── Cached model trainers ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _train_logistic(df):
    try:
        return run_logistic(df)
    except Exception as e:
        st.error(f"Logistic Regression error: {e}")
        return None

@st.cache_data(show_spinner=False)
def _train_svm(df):
    try:
        return run_svm(df)
    except Exception as e:
        st.error(f"SVM error: {e}")
        return None

@st.cache_data(show_spinner=False)
def _train_linear_regression(df):
    try:
        return run_linear_regression(df)
    except Exception as e:
        st.error(f"Linear Regression error: {e}")
        return None

@st.cache_data(show_spinner=False)
def _train_random_forest(df):
    try:
        return run_random_forest(df)
    except Exception as e:
        st.error(f"Random Forest error: {e}")
        return None

@st.cache_data(show_spinner=False)
def _predict_volatility(df):
    try:
        return predict_volatility(df)
    except Exception as e:
        st.error(f"Volatility prediction error: {e}")
        return None

# ── Reusable chart builders ───────────────────────────────────────────────────
def equity_curve_chart(bt: dict) -> go.Figure:
    ec = bt['equity_curve']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec['date'], y=ec['cum_asset'],
        name="Buy & Hold",
        line=dict(color="#8b949e", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=ec['date'], y=ec['cum_strategy'],
        name="Strategy",
        line=dict(color="#3fb950", width=2),
        fill='tonexty', fillcolor='rgba(63,185,80,0.06)',
    ))
    fig.update_layout(title="Equity Curve", **CHART_LAYOUT)
    return fig

def backtest_metrics_row(bt: dict):
    strat_ret = bt['total_strategy_return']
    asset_ret  = bt['total_asset_return']
    alpha      = strat_ret - asset_ret

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategy Return",  f"{strat_ret:.2%}")
    c2.metric("Buy & Hold Return", f"{asset_ret:.2%}")
    c3.metric("Alpha",  f"{alpha:+.2%}",
              delta_color="normal" if alpha >= 0 else "inverse")
    c4.metric("Sharpe Ratio",   f"{bt['sharpe']:.2f}")
    c5.metric("Max Drawdown",   f"{bt['max_drawdown']:.2%}")

def classification_report_table(report: dict):
    df = pd.DataFrame(report).transpose()
    df = df.round(3)
    st.dataframe(df, use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-label'>Configuration</div>", unsafe_allow_html=True)
    ticker     = st.text_input("Symbol", value="EURUSD").upper()
    end_date   = datetime.today()
    start_date = st.date_input("Start Date", value=end_date - timedelta(days=730))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Model Settings</div>", unsafe_allow_html=True)
    train_ratio = st.slider("Train / Test Split", 0.60, 0.90, TRAIN_RATIO, 0.05,
                            help="Fraction of data used for training")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    load_btn = st.button("⬇  Load Data", use_container_width=True)

    if load_btn:
        try:
            with st.spinner("Fetching & cleaning data…"):
                df_raw  = load_and_clean_data(ticker, start_date, end_date)
                df_feat = create_features(df_raw)
            st.session_state.update({
                'df_raw':  df_raw,
                'df_feat': df_feat,
                'ticker':  ticker,
                'train_ratio': train_ratio,
            })
            st.success(f"Loaded {len(df_raw):,} rows for {ticker}")
        except Exception as exc:
            st.error(f"Failed to load data: {exc}")

# ── Guard: no data yet ────────────────────────────────────────────────────────
if 'df_feat' not in st.session_state:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:60vh;gap:1rem;'>
        <span style='font-size:3rem;'>📡</span>
        <h2 style='color:#8b949e;font-weight:400;'>No data loaded</h2>
        <p style='color:#484f58;font-size:0.85rem;'>
            Enter a symbol and click <strong style='color:#58a6ff'>Load Data</strong> in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Unpack session state ──────────────────────────────────────────────────────
df_raw       = st.session_state['df_raw']
df_feat      = st.session_state['df_feat']
ticker       = st.session_state.get('ticker', 'SYMBOL')
train_ratio  = st.session_state.get('train_ratio', TRAIN_RATIO)

# Date column — support both column and index
date_col = df_raw['Date'] if 'Date' in df_raw.columns else df_raw.index
feat_date = df_feat['Date'] if 'Date' in df_feat.columns else df_feat.index

# Validate feature columns
missing = [f for f in FEATURES + ['target'] if f not in df_feat.columns]
if missing:
    st.error(f"Missing columns in feature data: {missing}")
    st.stop()

X = df_feat[FEATURES]
y = df_feat['target']

split_idx = int(len(df_feat) * train_ratio)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
price_test = df_feat['Close'].iloc[split_idx:].values
dates_test = np.array(feat_date[split_idx:])

# ── Header ────────────────────────────────────────────────────────────────────
c_title, c_badge = st.columns([6, 1])
with c_title:
    st.markdown(f"## {ticker} — Quant Validation Dashboard")
with c_badge:
    st.markdown(
        f"<div style='text-align:right;margin-top:0.6rem;"
        f"font-size:0.72rem;color:#58a6ff;'>"
        f"{start_date} → {end_date.strftime('%Y-%m-%d')}</div>",
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_names = [
    "📈 Overview",
    "🔵 Logistic Regression",
    "🟠 SVM",
    "� Linear Regression",
    "🌲 Random Forest",
    "📊 Volatility Prediction",
]
tabs = st.tabs(tab_names)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # Top metrics
    close_prices = df_raw['Close'] if 'Close' in df_raw.columns else df_raw['close']
    total_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
    daily_ret    = close_prices.pct_change().dropna()
    volatility   = daily_ret.std() * np.sqrt(252)
    sharpe_bh    = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Period Return",     f"{total_return:.2%}")
    m2.metric("Ann. Volatility",   f"{volatility:.2%}")
    m3.metric("Sharpe (B&H)",      f"{sharpe_bh:.2f}")
    m4.metric("Observations",      f"{len(df_raw):,}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Candlestick
    st.markdown("<div class='section-label'>Price Action</div>", unsafe_allow_html=True)
    o = df_raw['Open']  if 'Open'  in df_raw.columns else df_raw['open']
    h = df_raw['High']  if 'High'  in df_raw.columns else df_raw['high']
    l = df_raw['Low']   if 'Low'   in df_raw.columns else df_raw['low']
    c = df_raw['Close'] if 'Close' in df_raw.columns else df_raw['close']

    candle = go.Figure(go.Candlestick(
        x=date_col, open=o, high=h, low=l, close=c,
        increasing_line_color='#3fb950',
        decreasing_line_color='#f85149',
    ))
    candle.update_layout(xaxis_rangeslider_visible=False,
                         title=f"{ticker} OHLC", **CHART_LAYOUT)
    st.plotly_chart(candle, use_container_width=True)

    # Return distribution
    st.markdown("<div class='section-label'>Return Distribution</div>", unsafe_allow_html=True)
    ret_hist = px.histogram(
        daily_ret, nbins=80,
        labels={"value": "Daily Return"},
        color_discrete_sequence=["#58a6ff"],
    )
    ret_hist.update_layout(showlegend=False, title="Daily Returns", **CHART_LAYOUT)
    st.plotly_chart(ret_hist, use_container_width=True)

    # Feature correlation heatmap
    st.markdown("<div class='section-label'>Feature Correlation</div>", unsafe_allow_html=True)
    corr = X.corr()
    heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    heat.update_layout(title="Feature Correlation Matrix", **CHART_LAYOUT)
    st.plotly_chart(heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    with st.spinner("Training Logistic Regression…"):
        lr = _train_logistic(df_feat)
    
    if lr:
        # Metrics
        a1, a2, a3 = st.columns(3)
        a1.metric("Signal", f"{lr.get('signal', 'N/A')}")
        a2.metric("Probability Up", f"{lr.get('prob_up', 0):.2%}")
        a3.metric("Prediction", f"{lr.get('prediction', 'N/A')}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.info("✅ Logistic Regression trained on cleaned and engineered features from data_cleaner.py")
    else:
        st.error("Failed to train Logistic Regression model")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SVM
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    with st.spinner("Training SVM…"):
        svm = _train_svm(df_feat)
    
    if svm:
        # Metrics
        b1, b2, b3 = st.columns(3)
        b1.metric("Signal", f"{svm.get('signal', 'N/A')}")
        b2.metric("Probability Up", f"{svm.get('prob_up', 0):.2%}")
        b3.metric("Prediction", f"{svm.get('prediction', 'N/A')}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.info("✅ SVM trained on cleaned and engineered features from data_cleaner.py")
    else:
        st.error("Failed to train SVM model")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Linear Regression
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    with st.spinner("Training Linear Regression…"):
        lr_model = _train_linear_regression(df_feat)
    
    if lr_model:
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Signal", f"{lr_model.get('signal', 'N/A')}")
        c2.metric("Prediction", f"{lr_model.get('prediction', 'N/A')}")
        c3.metric("Model Type", "Regression")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.info("✅ Linear Regression trained on cleaned and engineered features")
    else:
        st.error("Failed to train Linear Regression model")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    with st.spinner("Training Random Forest…"):
        rf_model = _train_random_forest(df_feat)
    
    if rf_model:
        # Metrics
        d1, d2, d3 = st.columns(3)
        d1.metric("Signal", f"{rf_model.get('signal', 'N/A')}")
        d2.metric("Probability Up", f"{rf_model.get('prob_up', 0):.2%}")
        d3.metric("Prediction", f"{rf_model.get('prediction', 'N/A')}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.info("✅ Random Forest trained on cleaned and engineered features")
    else:
        st.error("Failed to train Random Forest model")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Volatility Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    with st.spinner("Predicting Volatility (GARCH)…"):
        vol_pred = _predict_volatility(df_feat)
    
    if vol_pred:
        # Metrics
        e1, e2, e3 = st.columns(3)
        e1.metric("Daily Volatility", f"{vol_pred.get('daily_volatility', 0):.2%}")
        e2.metric("Annual Volatility", f"{vol_pred.get('annual_volatility', 0):.2%}")
        e3.metric("Forecast Available", "Yes" if vol_pred.get('forecast') is not None else "No")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        if vol_pred.get('forecast') is not None:
            st.success("✅ GARCH(1,1) model successfully fitted with 5-day forecast")
            forecast_data = vol_pred.get('forecast')
            st.write(f"Volatility Forecast: {forecast_data}")
        else:
            st.warning("⚠️ Insufficient data for GARCH model (need >50 observations)")
    else:
        st.error("Failed to predict volatility")