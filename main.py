

# Pipeline controller
# This runs everything automatically.

from ingestion.mt5_download import download_data
from processing.data_cleaner import clean_data
from features.feature_engineering import add_features
from strategy.strategy import generate_signals
from backtest.backtest_engine import run_backtest
from backtest.performance import analyze_performance
from ml.svm_filter import apply_svm_filter
from ml.walk_forward import walk_forward_validation

symbols = ["EURUSD","GBPUSD","NZDUSD"]

for symbol in symbols:
    run_model(symbol)




