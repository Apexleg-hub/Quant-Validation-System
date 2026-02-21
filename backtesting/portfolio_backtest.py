

import pandas as pd

from strategy.atr_risk import apply_atr_risk
from strategy.svm_filter import svm_trade_filter
from backtest.performance import performance_report


ASSETS = ["EURUSD", "GBPUSD", "NZDUSD"]


class PortfolioBacktest:

    def __init__(self):

        self.weights = {
            "EURUSD": 0.33,
            "GBPUSD": 0.33,
            "NZDUSD": 0.34
        }

        self.data = {}


    def load_data(self):

        for asset in ASSETS:

            df = pd.read_csv(f"data/{asset}.csv")

            if 'Close' in df.columns:
                close_col = 'Close'
            elif 'close' in df.columns:
                close_col = 'close'
            else:
                raise KeyError(f"{asset} is missing close/Close column")

            df['Returns'] = df[close_col].pct_change()

            df = apply_atr_risk(df)

            df = svm_trade_filter(df)

            df['Strategy_Return'] = df['Returns'] * df['Signal']

            self.data[asset] = df


    def align_data(self):

        combined = pd.DataFrame()

        for asset in ASSETS:

            combined[asset] = self.data[asset]['Strategy_Return']

        combined.dropna(inplace=True)

        return combined


    def calculate_portfolio_returns(self, combined):

        portfolio_returns = pd.Series(0.0, index=combined.index)

        for asset in ASSETS:

            portfolio_returns += combined[asset] * self.weights[asset]

        return portfolio_returns
    
    def rebalance(self, combined):

        vol = combined.std()

        inv_vol = 1 / vol

        weights = inv_vol / inv_vol.sum()

        return weights



    def run(self):

        print("Running Portfolio Backtest...")

        self.load_data()

        combined = self.align_data()

        portfolio_returns = self.calculate_portfolio_returns(combined)

        equity = (1 + portfolio_returns).cumprod()

        return equity



if __name__ == "__main__":

    engine = PortfolioBacktest()

    equity = engine.run()

    performance_report(equity)
