

import pandas as pd
from processing.data_cleaner import clean_data, create_features
from strategy.atr_risk import apply_atr_risk
from strategy.svm_filter import svm_trade_filter
from backtest.performance import performance_report

ASSETS = ["EURUSD", "GBPUSD", "NZDUSD"]

class MultiAssetBacktest:

    def __init__(self, capital=10000):

        self.initial_capital = capital
        self.capital = capital
        self.results = []

    def load_data(self, asset):

        df = pd.read_csv(f"data/{asset}.csv")
        if 'Close' in df.columns:
            close_col = 'Close'
        elif 'close' in df.columns:
            close_col = 'close'
        else:
            raise KeyError(f"{asset} is missing close/Close column")
        df['Returns'] = df[close_col].pct_change()

        return df


    def run_asset(self, asset):

        print(f"\nRunning {asset}...")

        df = self.load_data(asset)

        df = apply_atr_risk(df)

        df = svm_trade_filter(df)

        df['Strategy_Return'] = df['Returns'] * df['Signal']

        return df


    def run(self):

        combined = pd.DataFrame()

        for asset in ASSETS:

            df = self.run_asset(asset)

            combined[asset] = df['Strategy_Return']

        combined = combined.dropna()
        combined_returns = combined.mean(axis=1)

        equity = (1 + combined_returns).cumprod()

        return equity


if __name__ == "__main__":

    engine = MultiAssetBacktest()

    equity = engine.run()

    performance_report(equity)
