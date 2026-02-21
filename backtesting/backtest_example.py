"""
Example Professional Backtest
Demonstrates complete workflow: data loading, signal generation, 
walk-forward validation, and result reporting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
from backtest_engine import BacktestEngine
from strategy_runner import StrategyRunner, SignalGenerator
from walk_forward import WalkForwardValidator


def load_sample_ohlcv_data(symbol: str, bars: int = 500) -> pd.DataFrame:
    """
    Generate sample OHLCV data for demonstration
    
    Args:
        symbol: Asset symbol
        bars: Number of bars to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    # Generate synthetic returns
    np.random.seed(hash(symbol) % 2**32)
    
    dates = pd.date_range(end=datetime.now(), periods=bars, freq='D')
    close_price = 100.0
    returns = np.random.normal(0.0005, 0.02, bars)
    
    prices = [close_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    df = pd.DataFrame({
        'open': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, bars),
    }, index=dates)
    
    return df


def prepare_signals(data: Dict[str, pd.DataFrame], strategy: str = 'sma') -> Dict[str, pd.DataFrame]:
    """
    Prepare data with trading signals
    
    Args:
        data: Dict of symbol -> OHLCV DataFrame
        strategy: Strategy name ('sma', 'rsi', 'bb', or 'combined')
        
    Returns:
        Dict of symbol -> DataFrame with added signal column
    """
    data_with_signals = {}
    
    for symbol, df in data.items():
        df = df.copy()
        
        if strategy == 'sma':
            df['signal'] = SignalGenerator.sma_crossover(df, fast=10, slow=20)
        elif strategy == 'rsi':
            df['signal'] = SignalGenerator.rsi_signal(df, period=14)
        elif strategy == 'bb':
            df['signal'] = SignalGenerator.bollinger_bands_signal(df, period=20)
        elif strategy == 'combined':
            signals = {
                'sma': SignalGenerator.sma_crossover(df, fast=10, slow=20),
                'rsi': SignalGenerator.rsi_signal(df, period=14),
                'bb': SignalGenerator.bollinger_bands_signal(df, period=20),
            }
            df['signal'] = SignalGenerator.combine_signals(signals, weights={'sma': 0.5, 'rsi': 0.3, 'bb': 0.2})
        
        data_with_signals[symbol] = df
    
    return data_with_signals


def backtest_single(data: Dict[str, pd.DataFrame], initial_capital: float = 100000) -> dict:
    """
    Run a single backtest on full dataset
    
    Args:
        data: Dict of symbol -> DataFrame with signals
        initial_capital: Starting capital
        
    Returns:
        Backtest results
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,  # 0.1%
        slippage_pcts={'EURUSD': 0.0001, 'GBPUSD': 0.0001},
        max_position_size=0.3,  # Max 30% per position
    )
    
    runner = StrategyRunner(
        backtest_engine=engine,
        data=data,
        entry_on='signal',
        use_close=True,
        max_trades_per_day=10,
        max_drawdown_limit_pct=15,
    )
    
    results = runner.run()
    return results


def backtest_walk_forward_fold(
    engine_class,
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    **kwargs
) -> tuple:
    """
    Run walk-forward fold backtest
    
    Args:
        engine_class: BacktestEngine class
        train_data: Training data (dict of symbol -> df)
        test_data: Out-of-sample test data (dict of symbol -> df)
        
    Returns:
        (is_stats, oos_stats)
    """
    # Add signals to both datasets
    train_with_signals = prepare_signals(train_data, strategy='combined')
    test_with_signals = prepare_signals(test_data, strategy='combined')
    
    # Run in-sample backtest
    is_results = backtest_single(train_with_signals, initial_capital=kwargs.get('initial_capital', 100000))
    is_stats = is_results['stats']
    
    # Run out-of-sample backtest
    oos_results = backtest_single(test_with_signals, initial_capital=kwargs.get('initial_capital', 100000))
    oos_stats = oos_results['stats']
    
    return is_stats, oos_stats


def main_simple_backtest():
    """Run a simple backtest on multiple assets"""
    print("=" * 80)
    print("PROFESSIONAL BACKTESTING FRAMEWORK - SIMPLE EXAMPLE")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    data = {symbol: load_sample_ohlcv_data(symbol, bars=500) for symbol in symbols}
    
    # Prepare signals
    print("[2] Generating trading signals...")
    data_with_signals = prepare_signals(data, strategy='combined')
    
    # Run backtest
    print("[3] Running backtest...")
    results = backtest_single(data_with_signals, initial_capital=100000)
    
    # Print results
    print_backtest_results(results)
    
    return results


def main_walk_forward():
    """Run walk-forward validation"""
    print("\n" + "=" * 80)
    print("PROFESSIONAL BACKTESTING FRAMEWORK - WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    symbols = ['EURUSD', 'GBPUSD']
    raw_data = {symbol: load_sample_ohlcv_data(symbol, bars=500) for symbol in symbols}
    
    # We'll use data from first symbol for WF split
    print("[2] Setting up walk-forward validation...")
    wf_validator = WalkForwardValidator(
        data=raw_data['EURUSD'],
        is_period=250,  # 1 year of daily data
        oos_period=50,  # 2 months test
        step_size=50,
    )
    
    print(f"   Generated {len(wf_validator.folds)} folds")
    
    # Execute walk-forward backtest
    print("[3] Running walk-forward backtest...")
    
    # Create custom split function
    def create_wf_strategy(engine_class, train_df, test_df, **kwargs):
        """Strategy runner for walk-forward fold"""
        # Create data dicts with multiple symbols
        train_data = {symbol: raw_data[symbol].loc[train_df.index] for symbol in raw_data}
        test_data = {symbol: raw_data[symbol].loc[test_df.index] for symbol in raw_data}
        
        return backtest_walk_forward_fold(engine_class, train_data, test_data, **kwargs)
    
    results = wf_validator.run(
        strategy_func=create_wf_strategy,
        backtest_engine_class=BacktestEngine,
        initial_capital=100000,
    )
    
    # Print summary
    wf_validator.print_summary(results)
    
    return results


def print_backtest_results(results: dict):
    """Pretty print backtest results"""
    stats = results['stats']
    trades_df = results['trades']
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        return
    
def run_backtest():
    """Run the backtest and print results"""
    results = main_simple_backtest()
    print_backtest_results(results)
    
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Total Return:        {stats.get('total_return_pct', 0):>10.2f}%")
    print(f"Annual Return:       {stats.get('annual_return_pct', 0):>10.2f}%")
    print(f"Max Drawdown:        {stats.get('max_drawdown_pct', 0):>10.2f}%")
    print(f"Sharpe Ratio:        {stats.get('sharpe_ratio', 0):>10.2f}")
    print(f"Sortino Ratio:       {stats.get('sortino_ratio', 0):>10.2f}")
    print(f"Calmar Ratio:        {stats.get('calmar_ratio', 0):>10.2f}")
    
    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades:        {stats.get('total_trades', 0):>10.0f}")
    print(f"Winning Trades:      {stats.get('winning_trades', 0):>10.0f}")
    print(f"Losing Trades:       {stats.get('losing_trades', 0):>10.0f}")
    print(f"Win Rate:            {stats.get('win_rate_pct', 0):>10.2f}%")
    print(f"Profit Factor:       {stats.get('profit_factor', 0):>10.2f}")
    print(f"Recovery Factor:     {stats.get('recovery_factor', 0):>10.2f}")
    
    print("\n--- P&L STATISTICS ---")
    print(f"Total P&L:           ${stats.get('total_pnl', 0):>10.2f}")
    print(f"Avg Trade P&L:       ${stats.get('avg_trade_pnl', 0):>10.2f}")
    print(f"Avg Winning Trade:   ${stats.get('avg_win_pnl', 0):>10.2f}")
    print(f"Avg Losing Trade:    ${stats.get('avg_loss_pnl', 0):>10.2f}")
    print(f"Total Commission:    ${stats.get('total_commission', 0):>10.2f}")
    print(f"Final Equity:        ${stats.get('final_equity', 0):>10.2f}")
    
    print("\n--- TRADE LOG (First 10 trades) ---")
    if len(trades_df) > 0:
        print(trades_df[['entry_time', 'symbol', 'direction', 'entry_price', 'exit_price', 'pnl', 'pnl_pct']].head(10).to_string(index=False))
    else:
        print("No trades executed")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Simple backtest example
    results_simple = main_simple_backtest()
    
    # Walk-forward validation example
    results_wf = main_walk_forward()
    
    print("\n✓ Backtesting complete!")
