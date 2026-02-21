"""
Professional Backtesting Engine for Multi-Asset Strategies
Supports position tracking, risk management, commissions, slippage, and comprehensive metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    size: float
    entry_commission: float
    exit_commission: float
    pnl: float
    pnl_pct: float
    duration_bars: int


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    size: float
    entry_commission: float
    current_price: float = None
    
    def unrealized_pnl(self, current_price: float, exit_commission: float = 0) -> float:
        """Calculate unrealized P&L"""
        if self.direction == 'LONG':
            return (current_price - self.entry_price) * self.size - self.entry_commission - exit_commission
        else:  # SHORT
            return (self.entry_price - current_price) * self.size - self.entry_commission - exit_commission


class BacktestEngine:
    """
    Core backtesting engine supporting:
    - Multi-bar OHLC processing
    - Long/Short positions
    - Position sizing & risk management
    - Commission & slippage modeling
    - Comprehensive trade tracking
    - Walk-forward validation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,  # 0.1% per trade
        slippage_pcts: Dict[str, float] = None,
        max_position_size: float = 0.5,  # Max 50% of capital per position
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pcts = slippage_pcts or {}
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # State tracking
        self.equity = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.equity_dates = []
        self.nav_history = []  # Portfolio net asset value over time
        
    def reset(self):
        """Reset engine to initial state"""
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.equity_dates = []
        self.nav_history = []
    
    def _apply_slippage(self, symbol: str, price: float, direction: str) -> float:
        """Apply slippage to entry/exit price"""
        slippage_pct = self.slippage_pcts.get(symbol, 0.0001)  # default 0.01%
        if direction == 'LONG':
            return price * (1 + slippage_pct)
        else:  # SHORT
            return price * (1 - slippage_pct)
    
    def _calculate_position_size(self, symbol: str, price: float, risk_amount: float) -> float:
        """Calculate position size based on risk/account size"""
        max_size = (self.equity * self.max_position_size) / price
        risk_size = risk_amount / price if risk_amount > 0 else max_size
        return min(risk_size, max_size)
    
    def enter_position(
        self,
        symbol: str,
        direction: str,
        price: float,
        timestamp: datetime,
        size: Optional[float] = None,
        risk_amount: Optional[float] = None,
    ) -> bool:
        """
        Enter a position (LONG or SHORT)
        
        Args:
            symbol: Asset symbol
            direction: 'LONG' or 'SHORT'
            price: Entry price
            timestamp: Entry datetime
            size: Fixed position size (optional)
            risk_amount: Risk amount to size position (optional)
            
        Returns:
            True if position entered, False otherwise
        """
        # Check if position already exists
        if symbol in self.positions:
            warnings.warn(f"Position {symbol} already exists, skipping entry")
            return False
        
        # Determine position size
        if size is None:
            if risk_amount is None:
                risk_amount = self.equity * 0.02  # Default 2% risk
            size = self._calculate_position_size(symbol, price, risk_amount)
        
        # Apply slippage
        actual_price = self._apply_slippage(symbol, price, direction)
        
        # Calculate commission
        position_value = size * actual_price
        entry_commission = position_value * self.commission
        total_cost = position_value + entry_commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            warnings.warn(f"Insufficient cash for {symbol}. Required: {total_cost:.2f}, Available: {self.cash:.2f}")
            return False
        
        # Create position
        pos = Position(
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=actual_price,
            size=size,
            entry_commission=entry_commission,
            current_price=actual_price
        )
        
        # Update state
        self.positions[symbol] = pos
        self.cash -= total_cost
        self.equity -= total_cost
        
        return True
    
    def exit_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        full_exit: bool = True,
        exit_amount: Optional[float] = None,
    ) -> Optional[Trade]:
        """
        Exit position (fully or partially)
        
        Args:
            symbol: Asset symbol
            price: Exit price
            timestamp: Exit datetime
            full_exit: Whether to exit entire position
            exit_amount: Amount to exit (for partial exits)
            
        Returns:
            Trade object if position exited, None otherwise
        """
        if symbol not in self.positions:
            warnings.warn(f"No position {symbol} to exit")
            return None
        
        pos = self.positions[symbol]
        
        # Apply slippage
        actual_price = self._apply_slippage(symbol, price, 'EXIT')
        
        # Exit size
        exit_size = pos.size if full_exit else (exit_amount or pos.size)
        if exit_size > pos.size:
            exit_size = pos.size
        
        # Calculate exit commission
        exit_value = exit_size * actual_price
        exit_commission = exit_value * self.commission
        
        # Calculate P&L
        if pos.direction == 'LONG':
            pnl = (actual_price - pos.entry_price) * exit_size - pos.entry_commission * (exit_size / pos.size) - exit_commission
        else:  # SHORT
            pnl = (pos.entry_price - actual_price) * exit_size - pos.entry_commission * (exit_size / pos.size) - exit_commission
        
        pnl_pct = (pnl / (pos.entry_price * exit_size)) * 100 if pos.entry_price > 0 else 0
        
        # Create trade record
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=actual_price,
            size=exit_size,
            entry_commission=pos.entry_commission * (exit_size / pos.size),
            exit_commission=exit_commission,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_bars=0  # Set during backtest loop
        )
        
        # Update cash: add back exit proceeds
        self.cash += exit_value - exit_commission
        self.equity += pnl
        
        # Update or close position
        if full_exit or exit_size >= pos.size:
            del self.positions[symbol]
        else:
            pos.size -= exit_size
            pos.entry_commission *= (pos.size / (pos.size + exit_size))
        
        self.trades.append(trade)
        return trade
    
    def update_equity(self, market_prices: Dict[str, float]):
        """Update portfolio equity based on current market prices"""
        self.equity = self.cash
        
        for symbol, pos in self.positions.items():
            current_price = market_prices.get(symbol, pos.current_price)
            if current_price:
                pos.current_price = current_price
                unrealized = pos.unrealized_pnl(current_price)
                self.equity += unrealized
        
        self.nav_history.append(self.equity)
    
    def record_equity_point(self, timestamp: datetime):
        """Record equity at current time"""
        self.equity_curve.append(self.equity)
        self.equity_dates.append(timestamp)
    
    def get_equity_series(self) -> pd.Series:
        """Get equity curve as pandas Series"""
        return pd.Series(self.equity_curve, index=self.equity_dates)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get completed trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        data = {
            'entry_time': [t.entry_time for t in self.trades],
            'exit_time': [t.exit_time for t in self.trades],
            'symbol': [t.symbol for t in self.trades],
            'direction': [t.direction for t in self.trades],
            'entry_price': [t.entry_price for t in self.trades],
            'exit_price': [t.exit_price for t in self.trades],
            'size': [t.size for t in self.trades],
            'commission': [t.entry_commission + t.exit_commission for t in self.trades],
            'pnl': [t.pnl for t in self.trades],
            'pnl_pct': [t.pnl_pct for t in self.trades],
            'duration_bars': [t.duration_bars for t in self.trades],
        }
        return pd.DataFrame(data)
    
    def get_open_positions_df(self) -> pd.DataFrame:
        """Get open positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        data = {
            'symbol': [p.symbol for p in self.positions.values()],
            'direction': [p.direction for p in self.positions.values()],
            'entry_price': [p.entry_price for p in self.positions.values()],
            'current_price': [p.current_price for p in self.positions.values()],
            'size': [p.size for p in self.positions.values()],
            'entry_time': [p.entry_time for p in self.positions.values()],
        }
        return pd.DataFrame(data)
    
    def close_all_positions(self, price_dict: Dict[str, float], timestamp: datetime):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            if symbol in price_dict:
                self.exit_position(symbol, price_dict[symbol], timestamp)
    
    def get_stats(self) -> Dict:
        """Calculate comprehensive backtest statistics"""
        if not self.trades:
            return {'error': 'No trades completed'}
        
        equity_series = self.get_equity_series()
        trades_df = self.get_trades_df()
        
        # Returns
        returns = equity_series.pct_change().dropna()
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100 if len(equity_series) > 0 else 0
        
        # Drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino = 0
        
        # Win Rate
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        total_trades = len(trades_df)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Average Trade
        avg_trade = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        
        # Calmar Ratio
        annual_return = total_return / max(1, len(equity_series) / 252)
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Recovery Factor
        total_pnl = trades_df['pnl'].sum()
        recovery_factor = total_pnl / abs(gross_loss) if gross_loss > 0 else 0
        
        # Duration stats
        avg_duration = trades_df['duration_bars'].mean() if len(trades_df) > 0 else 0
        
        return {
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'avg_trade_pnl': avg_trade,
            'avg_win_pnl': avg_win,
            'avg_loss_pnl': avg_loss,
            'total_pnl': total_pnl,
            'total_commission': trades_df['commission'].sum(),
            'avg_trade_duration_bars': avg_duration,
            'recovery_factor': recovery_factor,
            'final_equity': equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital,
        }
    def run_backtest(self, data: pd.DataFrame, strategy_func):
        """
        Run backtest on provided data using the given strategy function.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            strategy_func: Function that takes (engine, current_bar) and implements strategy logic
        """
        self.reset()
        
        for idx, row in data.iterrows():
            timestamp = idx
            market_prices = {col.split('_')[0]: row[col] for col in data.columns if '_' in col}
            
            # Update equity based on current market prices
            self.update_equity(market_prices)
            self.record_equity_point(timestamp)
            
            # Execute strategy logic for current bar
            strategy_func(self, row)
            
            # Update trade durations
            for trade in self.trades:
                if trade.exit_time is None:
                    trade.duration_bars += 1
        
        return self.get_stats()
