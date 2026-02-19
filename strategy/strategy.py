import pandas as pd

def generate_signals():

    df = pd.read_csv("data/features/XAUUSD_H1_features.csv")

    df['signal'] = 0

    buy = (df['EMA50'] > df['EMA200']) & (df['close'] > df['EMA200'])

    sell = (df['EMA50'] < df['EMA200']) & (df['close'] < df['EMA200'])

    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = -1


    # ATR filter
    df['ATR_mean'] = df['ATR'].rolling(50).mean()

    df.loc[df['ATR'] <= df['ATR_mean'], 'signal'] = 0


    # CREATE SL TP
    df['SL'] = None
    df['TP'] = None


    df.loc[df['signal'] == 1, 'SL'] = df['close'] - df['ATR'] * 2
    df.loc[df['signal'] == 1, 'TP'] = df['close'] + df['ATR'] * 4


    df.loc[df['signal'] == -1, 'SL'] = df['close'] + df['ATR'] * 2
    df.loc[df['signal'] == -1, 'TP'] = df['close'] - df['ATR'] * 4


    df.to_csv("data/features/XAUUSD_H1_signals.csv", index=False)

    print("Signals with SL TP generated")


generate_signals()
