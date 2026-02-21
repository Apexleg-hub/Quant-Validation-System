import MetaTrader5 as mt5
from sklearn.linear_model import LogisticRegression
from ingestion.mt5_download import download_data

symbols = ["EURUSD", "GBPUSD", "NZDUSD"]
results = {}

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

try:
    for symbol in symbols:
        df = download_data(symbol, "H4")

        # Compute features before resetting index
        pct_change = df["close"].pct_change() * 100
        tick_vol   = df["tick_volume"].shift(1) / 1_000_000  # correct column name

        df_filter = pct_change.rename("Today").reset_index()
        df_filter["Volume"] = tick_vol.values  # safe: same length, same order

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
        results[symbol] = accuracy
        print(f"{symbol}  Accuracy: {accuracy:.2%}")

finally:
    mt5.shutdown()

print("\nFinal results:", results)