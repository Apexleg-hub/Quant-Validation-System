# data_cleaner.py
# Standalone module: cleaning, feature engineering, model training, and prediction.
# Imported and used by mt5_download.py — do not run this file directly.

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ── Column requirements ───────────────────────────────────────────────────────

REQUIRED_COLS = ["open", "high", "low", "close","volume"]

# Features used for model training and prediction
FEATURE_COLS = [
    "sma_10", "sma_50",
    "roc_5", "roc_10",
    "hl_range",
    "rsi_14",
    "volatility",   # rolling std of returns — added by this module
]

# ── 0. Load & Clean Helper ────────────────────────────────────────────────────

def load_and_clean_data(symbol: str, start_date, end_date) -> pd.DataFrame:
    """
    Download data from MT5 and clean it.
    
    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "EURUSD")
    start_date : datetime or str
        Start date for data fetch
    end_date : datetime or str
        End date for data fetch
    
    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV data
    """
    try:
        # Import here to avoid circular imports
        from mt5_download import download_data
        
        # Download D1 timeframe data (adjust as needed)
        raw_data = download_data(symbol, "D1")
        
        # Filter by date range if dataframe has time index
        if hasattr(raw_data.index, 'date'):
            mask = (raw_data.index.date >= pd.to_datetime(start_date).date()) & \
                   (raw_data.index.date <= pd.to_datetime(end_date).date())
            raw_data = raw_data[mask]
        
        # Clean the data
        return clean_data(raw_data)
    except ImportError:
        # Fallback: if MT5 not available, raise clear error
        raise RuntimeError(
            f"Cannot download {symbol} data. Ensure MT5 is running and connected. "
            "Or provide pre-downloaded data directly."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load and clean data for {symbol}: {e}")

# ── 1. Clean ──────────────────────────────────────────────────────────────────

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitise raw OHLCV data from MT5."""
    data = data.copy()

    missing = [c for c in REQUIRED_COLS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows where close is NaN or non-positive
    data = data.dropna(subset=["close"])
    data = data[data["close"] > 0]

    return data

# ── 2. Feature Engineering ────────────────────────────────────────────────────

def create_features(data: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Add technical indicators and a binary target column.

    Features added
    --------------
    sma_10, sma_50   : simple moving averages
    roc_5, roc_10    : rate of change (momentum)
    hl_range         : intrabar volatility proxy (high-low / close)
    rsi_14           : relative strength index
    volatility       : rolling std of 1-bar returns (vol_window bars)
    return           : 1-bar percentage change (intermediate, not a feature)
    target           : 1 if next close > current close, else 0
    """
    data = data.copy()

    # Moving averages
    data["sma_10"] = data["close"].rolling(10).mean()
    data["sma_50"] = data["close"].rolling(50).mean()

    # Momentum
    data["roc_5"]  = data["close"].pct_change(5)
    data["roc_10"] = data["close"].pct_change(10)

    # Intrabar range
    data["hl_range"] = (data["high"] - data["low"]) / data["close"]

    # RSI (14)
    delta = data["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    # Return & rolling volatility
    data["return"]     = data["close"].pct_change()
    data["volatility"] = data["return"].rolling(vol_window).std()

    # Target: will next bar close higher?
    data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

    # Drop any rows that still have NaNs in features or target
    data = data.dropna(subset=FEATURE_COLS + ["target"])

    return data

# ── 3. Train ──────────────────────────────────────────────────────────────────

def train_model(data: pd.DataFrame):
    """
    Train a logistic regression pipeline on all rows except the last.

    Returns
    -------
    model : fitted sklearn Pipeline (StandardScaler + LogisticRegression)
    """
    train_data = data.iloc[:-1]   # last row has no future close to validate against

    X = train_data[FEATURE_COLS]
    y = train_data["target"]

    if X.empty or y.nunique() < 2:
        raise ValueError("Not enough valid data or only one class present — cannot train.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(X, y)
    return model

# ── 4. Predict ────────────────────────────────────────────────────────────────

def predict(data: pd.DataFrame, model) -> dict:
    """
    Predict direction for the most recent bar.

    Returns
    -------
    dict with keys:
        signal     : "BUY", "SELL", or "HOLD" (HOLD when confidence < 55 %)
        prediction : raw int  (1 = up, 0 = down)
        prob_up    : probability of an up move (float 0–1)
    """
    latest = data[FEATURE_COLS].dropna().iloc[[-1]]  # shape (1, n_features)

    raw_pred = int(model.predict(latest)[0])
    prob_up  = float(model.predict_proba(latest)[0, 1])
    confidence = prob_up if raw_pred == 1 else (1 - prob_up)

    if confidence < 0.55:
        signal = "HOLD"
    else:
        signal = "BUY" if raw_pred == 1 else "SELL"

    return {
        "signal":     signal,
        "prediction": raw_pred,
        "prob_up":    prob_up,
    }