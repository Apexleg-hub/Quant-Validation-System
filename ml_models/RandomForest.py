import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from ingestion.data_cleaner import clean_data, create_features, train_model, predict

def run_random_forest(df):
    """
    Train random forest model on clean data with engineered features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    
    Returns
    -------
    dict
        Contains keys: model, signal, prob_up, prediction, feature_importances
    """
    try:
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
        raise RuntimeError(f"Random forest training failed: {e}")