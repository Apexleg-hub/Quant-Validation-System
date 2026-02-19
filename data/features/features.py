

import ta

def add_features():

    df = pd.read_csv("data/clean/XAUUSD_H1_clean.csv")

    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)

    df['EMA200'] = ta.trend.ema_indicator(df['close'], window=200)

    df['ATR'] = ta.volatility.average_true_range(
        df['high'],
        df['low'],
        df['close']
    )

    # ADD THIS

    df['ADX'] = ta.trend.adx(
        df['high'],
        df['low'],
        df['close'],
        window=14
    )

    df.to_csv("data/features/XAUUSD_H1_features.csv", index=False)
if __name__ == "__main__":
    add_features()
    