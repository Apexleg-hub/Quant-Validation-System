"""
Walk-Forward Validation Framework
Enables proper out-of-sample testing and model robustness validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WalkForwardFold:
    """Represents one fold in walk-forward analysis"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    is_oos: bool = True


class WalkForwardValidator:
    """
    Walk-Forward validation framework for robust strategy testing.
    Ensures no lookahead bias and proper out-of-sample validation.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        is_period: int = 252,  # In-sample period (bars)
        oos_period: int = 63,  # Out-of-sample period (bars)
        step_size: Optional[int] = None,  # Step forward each fold
    ):
        """
        Initialize walk-forward validator
        
        Args:
            data: DataFrame with datetime index and OHLCV columns
            is_period: Number of bars for in-sample training
            oos_period: Number of bars for out-of-sample testing
            step_size: How many bars to step forward each fold (default: oos_period)
        """
        self.data = data.copy()
        self.is_period = is_period
        self.oos_period = oos_period
        self.step_size = step_size or oos_period
        self.folds: List[WalkForwardFold] = []
        
        self._generate_folds()
    
    def _generate_folds(self):
        """Generate walk-forward folds"""
        self.folds = []
        fold_id = 0
        
        total_bars = len(self.data)
        min_required = self.is_period + self.oos_period
        
        if total_bars < min_required:
            raise ValueError(f"Not enough data. Need at least {min_required} bars, got {total_bars}")
        
        # Generate folds
        pos = 0
        while pos + min_required <= total_bars:
            train_start_idx = pos
            train_end_idx = pos + self.is_period
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.oos_period, total_bars)
            
            train_data = self.data.iloc[train_start_idx:train_end_idx]
            test_data = self.data.iloc[test_start_idx:test_end_idx]
            
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_data=train_data,
                test_data=test_data,
                is_oos=True
            )
            self.folds.append(fold)
            fold_id += 1
            pos += self.step_size
        
        if len(self.folds) == 0:
            raise ValueError("No folds generated. Check is_period and oos_period values.")
    
    def run(
        self,
        strategy_func: Callable,
        backtest_engine_class,
        **kwargs
    ) -> Dict:
        """
        Run walk-forward backtest
        
        Args:
            strategy_func: Function that takes (backtest_engine, train_data, test_data, **kwargs)
                          and returns (is_stats, oos_stats)
            backtest_engine_class: BacktestEngine class to use
            **kwargs: Additional arguments passed to strategy_func
            
        Returns:
            Dictionary with results for each fold
        """
        results = {
            'folds': [],
            'summary': {}
        }
        
        for fold in self.folds:
            print(f"\nRunning WF Fold {fold.fold_id + 1}/{len(self.folds)} "
                  f"({fold.test_start.date()} to {fold.test_end.date()})")
            
            # Run strategy on this fold
            is_stats, oos_stats = strategy_func(
                backtest_engine_class,
                fold.train_data,
                fold.test_data,
                **kwargs
            )
            
            fold_result = {
                'fold_id': fold.fold_id,
                'train_period': f"{fold.train_start.date()} to {fold.train_end.date()}",
                'test_period': f"{fold.test_start.date()} to {fold.test_end.date()}",
                'is_stats': is_stats,
                'oos_stats': oos_stats,
            }
            results['folds'].append(fold_result)
        
        # Aggregate results
        results['summary'] = self._aggregate_results(results['folds'])
        
        return results
    
    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate results across all folds"""
        is_metrics = {}
        oos_metrics = {}
        
        for fold in fold_results:
            for key, value in fold['is_stats'].items():
                if isinstance(value, (int, float)):
                    if key not in is_metrics:
                        is_metrics[key] = []
                    is_metrics[key].append(value)
            
            for key, value in fold['oos_stats'].items():
                if isinstance(value, (int, float)):
                    if key not in oos_metrics:
                        oos_metrics[key] = []
                    oos_metrics[key].append(value)
        
        # Calculate means and std devs
        summary = {}
        for key in is_metrics:
            if key in oos_metrics:
                is_vals = np.array(is_metrics[key])
                oos_vals = np.array(oos_metrics[key])
                
                summary[key] = {
                    'is_mean': np.mean(is_vals),
                    'is_std': np.std(is_vals),
                    'oos_mean': np.mean(oos_vals),
                    'oos_std': np.std(oos_vals),
                    'degradation': np.mean(is_vals) - np.mean(oos_vals),  # How much OOS underperforms IS
                    'efficiency': (np.mean(oos_vals) / np.mean(is_vals) * 100) if np.mean(is_vals) != 0 else 0,
                }
        
        return summary
    
    def print_summary(self, results: Dict):
        """Print walk-forward results summary"""
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*80)
        
        summary = results['summary']
        
        print("\nKey Metrics (IS vs OOS):")
        print("-" * 80)
        print(f"{'Metric':<25} {'IS Mean':<15} {'OOS Mean':<15} {'Efficiency':<15}")
        print("-" * 80)
        
        # Focus on key metrics
        key_metrics = ['total_return_pct', 'sharpe_ratio', 'win_rate_pct', 'max_drawdown_pct']
        
        for metric in key_metrics:
            if metric in summary:
                stats = summary[metric]
                is_mean = stats['is_mean']
                oos_mean = stats['oos_mean']
                eff = stats['efficiency']
                
                print(f"{metric:<25} {is_mean:>14.2f} {oos_mean:>14.2f} {eff:>13.1f}%")
        
        # Fold-by-fold results
        print("\n" + "="*80)
        print("FOLD-BY-FOLD RESULTS")
        print("="*80)
        
        for fold in results['folds']:
            print(f"\nFold {fold['fold_id'] + 1}: {fold['test_period']}")
            print(f"  IS Sharpe: {fold['is_stats'].get('sharpe_ratio', 0):.2f}")
            print(f"  OOS Sharpe: {fold['oos_stats'].get('sharpe_ratio', 0):.2f}")
            print(f"  IS Return: {fold['is_stats'].get('total_return_pct', 0):.2f}%")
            print(f"  OOS Return: {fold['oos_stats'].get('total_return_pct', 0):.2f}%")


class TimeSeriesSplitter:
    """
    Simple time-series train/test splitter to avoid lookahead bias
    """
    
    @staticmethod
    def split_by_date(
        data: pd.DataFrame,
        train_end_date: datetime,
        test_start_date: Optional[datetime] = None,
        test_end_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by date boundaries
        
        Args:
            data: DataFrame with datetime index
            train_end_date: Last date for training
            test_start_date: First date for testing
            test_end_date: Last date for testing
            
        Returns:
            (train_data, test_data)
        """
        train = data[data.index <= train_end_date]
        
        if test_start_date:
            test = data[(data.index > train_end_date) & (data.index >= test_start_date)]
        else:
            test = data[data.index > train_end_date]
        
        if test_end_date:
            test = test[test.index <= test_end_date]
        
        return train, test
    
    @staticmethod
    def split_by_pct(
        data: pd.DataFrame,
        train_pct: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by percentage
        
        Args:
            data: DataFrame with datetime index
            train_pct: Percentage of data for training (0.0-1.0)
            
        Returns:
            (train_data, test_data)
        """
        split_idx = int(len(data) * train_pct)
        return data.iloc[:split_idx], data.iloc[split_idx:]
