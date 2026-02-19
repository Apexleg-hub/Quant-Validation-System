

import MetaTrader5 as mt5
import pandas as pd


def download_data(symbol):

    rates = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_H1,
        0,
        5000
    )

    df = pd.DataFrame(rates)

    df['time'] = pd.to_datetime(df['time'], unit='s')

    return df
