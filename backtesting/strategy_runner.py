"""
Strategy Runner and Backtester
Executes trading strategies using the BacktestEngine with proper signal handling
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple, List
import warnings

from backtest_engine import BacktestEngine


class StrategyRunner:
    """
    Executes a strategy on historical data using the backtesting engine.
    Handles signal generation, position management, and risk limits.
    """
    
    def __init__(
        self,
        backtest_engine: BacktestEngine,
        data: Dict[str, pd.DataFrame],  # {symbol: ohlcv_df}
        signal_columns: Dict[str, str] = None,  # {symbol: 'signal_col_name'}
        entry_on: str = 'signal',  # Entry signal column name
        exit_on: Optional[str] = None,  # Exit signal column name
        use_close: bool = True,  # Use close price for entry/exit
        max_trades_per_day: int = None,
        max_drawdown_limit_pct: float = None,
    ):
        """
        Initialize strategy runner
        
        Args:
            backtest_engine: Initialized BacktestEngine
            data: Dict of symbol -> OHLCV DataFrame with signal columns
            signal_columns: Column names containing signals (default: uses entry_on)
            entry_on: Column name for entry signals (1/-1)
            exit_on: Column name for exit signals
            use_close: Use close price instead of open for entry/exit
            max_trades_per_day: Maximum trades allowed per day
            max_drawdown_limit_pct: Stop trading if drawdown exceeds limit
        """
        self.engine = backtest_engine
        self.data = data
        self.entry_on = entry_on
        self.exit_on = exit_on
        self.use_close = use_close
        self.max_trades_per_day = max_trades_per_day
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        
        # Align all dataframes to same index
        self._align_data()
    
    def _align_data(self):
        """Align all dataframes to common index"""
        if not self.data:
            return
        
        # Get all indices
        indices = [df.index for df in self.data.values()]
        
        # Find common index
        all_dates = set(indices[0])
        for idx in indices[1:]:
            all_dates = all_dates.intersection(set(idx))
        
        # Reindex all data
        common_index = sorted(list(all_dates))
        for symbol in self.data:
            self.data[symbol] = self.data[symbol].loc[common_index]
    
    def run(self) -> Dict:
        """
        Run the strategy backtest
        
        Returns:
            Dictionary with results
        """
        self.engine.reset()
        
        # Get aligned dates across all symbols
        dates = None
        valid_symbols = {}
        
        for symbol, df in self.data.items():
            if self.entry_on in df.columns:
                valid_symbols[symbol] = df
                if dates is None:
                    dates = set(df.index)
                else:
                    dates = dates.intersection(set(df.index))
        
        if not valid_symbols or not dates:
            raise ValueError(f"No data with '{self.entry_on}' column or no common dates")
        
        dates = sorted(list(dates))
        trades_today = {}
        
        for bar_idx, timestamp in enumerate(dates):
            # Check drawdown limit
            if self.max_drawdown_limit_pct:
                equity_series = self.engine.get_equity_series()
                if len(equity_series) > 1:
                    cummax = equity_series.cummax()
                    drawdown = (self.engine.equity - cummax.iloc[-1]) / cummax.iloc[-1]
                    if drawdown < -self.max_drawdown_limit_pct / 100:
                        continue
            
            # Reset trade counter daily
            date = timestamp.date()
            if date not in trades_today:
                trades_today[date] = 0
            
            # Get current prices
            current_prices = {}
            for symbol, df in valid_symbols.items():
                current_prices[symbol] = df.loc[timestamp, 'close']
            
            # Update drawdown/equity
            self.engine.update_equity(current_prices)
            
            # Process symbols for exits first
            if self.exit_on:
                for symbol in list(self.engine.positions.keys()):
                    if symbol in valid_symbols:
                        row = valid_symbols[symbol].loc[timestamp]
                        if row.get(self.exit_on, 0) != 0:
                            price = row['close'] if self.use_close else row.get('open', row['close'])
                            trade = self.engine.exit_position(symbol, price, timestamp)
                            if trade:
                                trade.duration_bars = bar_idx - self._get_entry_bar(symbol)
            
            # Process symbols for entries
            for symbol, df in valid_symbols.items():
                # Skip if position exists
                if symbol in self.engine.positions:
                    continue
                
                # Check trade limit
                if self.max_trades_per_day and trades_today[date] >= self.max_trades_per_day:
                    continue
                
                row = df.loc[timestamp]
                signal = row.get(self.entry_on, 0)
                
                if signal != 0:
                    price = row['close'] if self.use_close else row.get('open', row['close'])
                    direction = 'LONG' if signal > 0 else 'SHORT'
                    
                    if self.engine.enter_position(
                        symbol=symbol,
                        direction=direction,
                        price=price,
                        timestamp=timestamp,
                        risk_amount=self.engine.equity * 0.02  # 2% risk per trade
                    ):
                        trades_today[date] += 1
                        self._store_entry_bar(symbol, bar_idx)
            
            # Record equity
            self.engine.record_equity_point(timestamp)
        
        # Close all positions at end
        final_prices = {}
        for symbol, df in valid_symbols.items():
            final_prices[symbol] = df.iloc[-1]['close']
        self.engine.close_all_positions(final_prices, dates[-1])
        
        return {
            'stats': self.engine.get_stats(),
            'trades': self.engine.get_trades_df(),
            'open_positions': self.engine.get_open_positions_df(),
            'equity_curve': self.engine.get_equity_series(),
        }
    
    def _exit_all_positions(self, timestamp: datetime, data: pd.DataFrame, bar_idx: int, reason: str):
        """Force exit all positions due to risk limit or other reason"""
        prices = {}
        for symbol in self.engine.positions:
            if symbol in data.columns:
                row = data[symbol].iloc[bar_idx]
                prices[symbol] = row['close']
        
        for symbol in list(self.engine.positions.keys()):
            if symbol in prices:
                trade = self.engine.exit_position(symbol, prices[symbol], timestamp)
                if trade:
                    trade.duration_bars = bar_idx - self._get_entry_bar(symbol)
    
    # Helper methods
    _entry_bars = {}
    
    def _store_entry_bar(self, symbol: str, bar_idx: int):
        """Store entry bar index for duration tracking"""
        self._entry_bars[symbol] = bar_idx
    
    def _get_entry_bar(self, symbol: str) -> int:
        """Get entry bar index"""
        return self._entry_bars.get(symbol, 0)


class SignalGenerator:
    """
    Generates trading signals from indicators
    """
    
    @staticmethod
    def sma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.Series:
        """
        Simple Moving Average crossover signals
        
        Args:
            df: DataFrame with 'close' column
            fast: Fast SMA period
            slow: Slow SMA period
            
        Returns:
            Signal series (1 for long, -1 for short, 0 for no signal)
        """
        sma_fast = df['close'].rolling(window=fast).mean()
        sma_slow = df['close'].rolling(window=slow).mean()
        
        signal = pd.Series(0, index=df.index)
        signal[sma_fast > sma_slow] = 1
        signal[sma_fast < sma_slow] = -1
        
        return signal
    
    @staticmethod
    def rsi_signal(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
        """
        RSI-based signals
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
            
        Returns:
            Signal series (1 for long, -1 for short, 0 for no signal)
        """
        close = df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signal = pd.Series(0, index=df.index)
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = -1
        
        return signal
    
    @staticmethod
    def bollinger_bands_signal(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger Bands mean reversion signals
        
        Args:
            df: DataFrame with 'close' column
            period: BB period
            std_dev: Number of standard deviations
            
        Returns:
            Signal series (1 for long, -1 for short, 0 for no signal)
        """
        close = df['close']
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        signal = pd.Series(0, index=df.index)
        signal[close < lower_band] = 1   # Mean reversion long
        signal[close > upper_band] = -1  # Mean reversion short
        
        return signal
    
    @staticmethod
    def combine_signals(signals: Dict[str, pd.Series], weights: Dict[str, float] = None) -> pd.Series:
        """
        Combine multiple signals with equal or weighted voting
        
        Args:
            signals: Dict of signal name -> signal series
            weights: Dict of signal name -> weight (default: equal weights)
            
        Returns:
            Combined signal series
        """
        if not signals:
            return pd.Series(0)
        
        if weights is None:
            weights = {name: 1.0 / len(signals) for name in signals}
        
        combined = pd.Series(0.0, index=list(signals.values())[0].index)
        
        for name, signal in signals.items():
            weight = weights.get(name, 1.0 / len(signals))
            combined += signal * weight
        
        # Convert to -1, 0, 1
        result = pd.Series(0, index=combined.index)
        result[combined > 0.5] = 1
        result[combined < -0.5] = -1
        
        return result
