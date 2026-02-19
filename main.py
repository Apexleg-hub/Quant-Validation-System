

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




def run_pipeline():

    print("STEP 1: Downloading data...")
    download_data()

    print("STEP 2: Cleaning data...")
    clean_data()

    print("STEP 3: Adding features...")
    add_features()

    print("STEP 4: Generating signals...")
    generate_signals()

    print("STEP 5: Applying SVM filter...")
    apply_svm_filter()
    

    print("STEP 6: Running backtest...")
    run_backtest()

    
    print("STEP 7: Walk-forward validation...")
    walk_forward_validation()

    print("STEP 8: Performance analysis...")
    analyze_performance()
    



    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    run_pipeline()
