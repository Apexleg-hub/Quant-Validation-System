import MetaTrader5 as mt5
from sklearn.linear_model import LogisticRegression
from ingestion.mt5_download import download_data

symbols = ["EURUSD", "GBPUSD", "NZDUSD"]
results = {}

BUY_THRESHOLD     = 0.60  # prob_up above this → BUY
SELL_THRESHOLD    = 0.40  # prob_up below this → SELL
                          # between the two    → NEUTRAL

def get_signal(prob_up: float) -> str:
    if prob_up >= BUY_THRESHOLD:
        return "BUY"
    elif prob_up <= SELL_THRESHOLD:
        return "SELL"
    else:
        return "NEUTRAL"

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

try:
    print(f"\n{'Symbol':<10} {'Accuracy':<12} {'Signal':<10} {'Prob Up':<10} {'Prob Down'}")
    print("-" * 55)

    for symbol in symbols:
        df = download_data(symbol, "H4")

        pct_change = df["close"].pct_change() * 100
        tick_vol   = df["tick_volume"].shift(1) / 1_000_000

        df_filter = pct_change.rename("Today").reset_index()
        df_filter["Volume"] = tick_vol.values

        for i in range(1, 6):
            df_filter[f"Lag {i}"] = df_filter["Today"].shift(i)

        df_filter = df_filter.dropna()
        df_filter["Direction"] = (df_filter["Today"] > 0).astype(int)

        X = df_filter[["Lag 1", "Lag 2", "Lag 3", "Lag 4", "Lag 5", "Volume"]]
        y = df_filter["Direction"]

        split = int(len(df_filter) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        # ── Confidence on the LATEST bar ──────────────────────────────
        latest_X   = X.iloc[[-1]]             # most recent row, shape (1, 6)
        proba      = model.predict_proba(latest_X)[0]  # [prob_down, prob_up]
        prob_down  = proba[0]
        prob_up    = proba[1]
        signal     = get_signal(prob_up)

        results[symbol] = {
            "accuracy":  accuracy,
            "signal":    signal,
            "prob_up":   prob_up,
            "prob_down": prob_down,
        }

        print(
            f"{symbol:<10} {accuracy:<12.2%} {signal:<10} "
            f"{prob_up:<10.2%} {prob_down:.2%}"
        )

finally:
    mt5.shutdown()

print("\nFinal results:")
for symbol, r in results.items():
    print(f"  {symbol}: {r['signal']} (up={r['prob_up']:.2%}, down={r['prob_down']:.2%}, acc={r['accuracy']:.2%})")
