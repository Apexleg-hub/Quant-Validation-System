"""
Logistic Regression Model - Uses data_cleaner for data processing.

This module provides logistic regression trading signals by leveraging
the data cleaning and feature generation from data_cleaner.py.
"""

from sklearn.linear_model import LogisticRegression
from ingestion.data_cleaner import clean_data, create_features, train_model, predict

BUY_THRESHOLD = 0.60  # prob_up above this → BUY
SELL_THRESHOLD = 0.40  # prob_up below this → SELL
                       # between the two    → NEUTRAL


def get_signal(prob_up: float) -> str:
    """Convert probability to trading signal."""
    if prob_up >= BUY_THRESHOLD:
        return "BUY"
    elif prob_up <= SELL_THRESHOLD:
        return "SELL"
    else:
        return "NEUTRAL"


def run_logistic(df):
    """
    Train logistic regression model on clean data with engineered features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    
    Returns
    -------
    dict
        Contains keys: model, signal, prob_up, prediction, accuracy, report
    """
    try:
        # Data pipeline: clean → features → train → predict
        df_clean = clean_data(df)
        df_features = create_features(df_clean)
        model = train_model(df_features)
        result = predict(df_features, model)
        
        return {
            'model': model,
            'signal': result['signal'],
            'prob_up': result['prob_up'],
            'prediction': result['prediction']
        }
    except Exception as e:
        raise RuntimeError(f"Logistic regression training failed: {e}")






