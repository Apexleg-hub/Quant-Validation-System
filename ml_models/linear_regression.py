import pandas as pd
import MetaTrader5 as mt5
from ingestion.mt5_download import download_data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

symbols = ["EURUSD", "GBPUSD", "NZDUSD"]
results = {}

THRESHOLD = 0.05  # % move needed to trigger BUY or SELL (tune this)

def get_signal(predicted_return_pct: float) -> str:
    if predicted_return_pct > THRESHOLD:
        return "BUY"
    elif predicted_return_pct < -THRESHOLD:
        return "SELL"
    else:
        return "NEUTRAL"

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

try:
    print(f"\n{'Symbol':<10} {'R²':<10} {'Signal':<10} {'Predicted Return'}")
    print("-" * 45)

    for symbol in symbols:
        df = download_data(symbol, "H4")

        data = df[["close"]].copy()
        data["next_close"] = data["close"].shift(-1)
        data.dropna(inplace=True)

        X = data[["close"]].values
        y = data["next_close"].values

        split = int(len(data) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2     = r2_score(y_test, y_pred)

        # ── Signal from the latest bar ────────────────────────────────
        current_close    = data["close"].iloc[-1]
        latest_X         = [[current_close]]
        predicted_close  = model.predict(latest_X)[0]
        predicted_return = (predicted_close - current_close) / current_close * 100
        signal           = get_signal(predicted_return)

        results[symbol] = {
            "r2":               r2,
            "signal":           signal,
            "predicted_return": predicted_return,
            "current_close":    current_close,
            "predicted_close":  predicted_close,
        }

        arrow = "▲" if signal == "BUY" else "▼" if signal == "SELL" else "–"
        print(
            f"{symbol:<10} {r2:<10.4f} {signal:<10} "
            f"{arrow} {predicted_return:+.4f}%"
        )

    # ── Detailed summary ──────────────────────────────────────────────
    print("\nDetailed breakdown:")
    print("-" * 55)
    for symbol, r in results.items():
        print(
            f"  {symbol}: {r['signal']}\n"
            f"    Current close:   {r['current_close']:.5f}\n"
            f"    Predicted close: {r['predicted_close']:.5f}\n"
            f"    Predicted move:  {r['predicted_return']:+.4f}%\n"
            f"    R²:              {r['r2']:.4f}\n"
        )

finally:
    mt5.shutdown()
