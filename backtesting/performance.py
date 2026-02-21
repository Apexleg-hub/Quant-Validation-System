import numpy as np
import pandas as pd


def performance_report(equity):
    if isinstance(equity, pd.Series):
        balance = equity.dropna().astype(float)
    else:
        eq = pd.DataFrame(equity).copy()
        if "balance" in eq.columns:
            balance = eq["balance"].dropna().astype(float)
        else:
            balance = eq.iloc[:, 0].dropna().astype(float)

    if balance.empty:
        print("No equity data to evaluate.")
        return

    returns = balance.pct_change().dropna()
    max_dd = ((balance / balance.cummax()) - 1.0).min() * 100
    sharpe = 0 if returns.std() == 0 else np.sqrt(252) * returns.mean() / returns.std()

    print("\n=== PERFORMANCE REPORT ===\n")
    print(f"Final Equity: {balance.iloc[-1]:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")


def analyze_performance():
    trades = pd.read_csv("backtest/trade_log.csv")
    equity = pd.read_csv("backtest/backtest_result.csv")

    print("\n=== EQUALLINE PERFORMANCE REPORT ===\n")

    win_rate = calculate_win_rate(trades)
    max_dd = calculate_max_drawdown(equity)
    sharpe = calculate_sharpe_ratio(equity)
    profit_factor = calculate_profit_factor(trades)

    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")

    evaluate_strategy(win_rate, max_dd, sharpe, profit_factor)


def calculate_win_rate(trades):
    wins = trades[trades["Profit"] > 0]
    total = len(trades)
    if total == 0:
        return 0
    return (len(wins) / total) * 100


def calculate_max_drawdown(equity):
    equity = equity.copy()
    equity["peak"] = equity["balance"].cummax()
    equity["drawdown"] = (equity["balance"] - equity["peak"]) / equity["peak"]
    return equity["drawdown"].min() * 100


def calculate_sharpe_ratio(equity):
    returns = equity["balance"].pct_change().dropna()
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * returns.mean() / returns.std()


def calculate_profit_factor(trades):
    profit = trades[trades["Profit"] > 0]["Profit"].sum()
    loss = abs(trades[trades["Profit"] < 0]["Profit"].sum())
    if loss == 0:
        return 0
    return profit / loss


def evaluate_strategy(win_rate, max_dd, sharpe, profit_factor):
    print("\n=== STRATEGY EVALUATION ===\n")

    if sharpe > 1.5 and profit_factor > 1.5 and max_dd > -20:
        print("STATUS: STRONG strategy")
    elif sharpe > 1 and profit_factor > 1:
        print("STATUS: ACCEPTABLE strategy")
    else:
        print("STATUS: WEAK strategy - needs improvement")


if __name__ == "__main__":
    analyze_performance()
