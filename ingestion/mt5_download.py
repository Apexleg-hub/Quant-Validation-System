# mt5_download.py
# Connects to MetaTrader 5, downloads OHLCV data, then delegates all
# feature engineering, training, and prediction to data_cleaner.py.
#
# Place both files in the same folder (e.g. ingestion/).
# Run from anywhere:  python ingestion/mt5_download.py

import sys
import os

# Ensure the folder this file lives in is always on sys.path,
# so `from data_cleaner import ...` works no matter where you
# launch Python from (project root, ingestion/, anywhere).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import pandas as pd

from data_cleaner import clean_data, create_features, train_model, predict

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]

TIMEFRAME_MAP = {
    
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "WK": mt5.TIMEFRAME_W1,
    "MN": mt5.TIMEFRAME_MN1,
}

BARS = 2000  # number of historical bars to fetch per symbol/timeframe

# ── Download ──────────────────────────────────────────────────────────────────

def download_data(symbol: str, timeframe_str: str) -> pd.DataFrame:
    """Fetch OHLCV bars from MT5 and return as a time-indexed DataFrame."""
    mt5_timeframe = TIMEFRAME_MAP.get(timeframe_str)
    if mt5_timeframe is None:
        raise ValueError(
            f"Invalid timeframe '{timeframe_str}'. "
            f"Valid options: {list(TIMEFRAME_MAP.keys())}"
        )

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, BARS)

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No data returned for {symbol} {timeframe_str}. "
            f"MT5 error: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df

# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(symbol: str, timeframe_str: str) -> None:
    """Full pipeline: download → clean → features → train → predict → print."""
    try:
        df     = download_data(symbol, timeframe_str)
        df     = clean_data(df)
        df     = create_features(df)
        model  = train_model(df)
        result = predict(df, model)

        print(
            f"{symbol:<10} {timeframe_str:<4}  →  "
            f"{result['signal']:<4}  "
            f"(prob_up={result['prob_up']:.2%})"
        )

    except Exception as exc:
        print(f"{symbol:<10} {timeframe_str:<4}  →  ERROR: {exc}")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed — error: {mt5.last_error()}")

    try:
        print(f"\n{'Symbol':<10} {'TF':<4}     Signal     Confidence")
        print("-" * 45)
        for symbol in SYMBOLS:
            for tf_name in TIMEFRAME_MAP:
                run_pipeline(symbol, tf_name)
    finally:
        mt5.shutdown()
        print("\nMT5 connection closed.")