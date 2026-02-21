import pandas as pd
from sklearn.linear_model import LinearRegression
from ingestion.data_cleaner import clean_data, create_features, train_model, predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

symbols = ["EURUSD", "GBPUSD", "NZDUSD"]
results = {}

THRESHOLD = 0.05  # % move needed to trigger BUY or SELL (tune this)

def get_signal(predicted_return_pct: float) -> str:
    if predicted_return_pct > THRESHOLD:
        return "BUY"
    elif predicted_return_pct < -THRESHOLD:
        return "SELL"
    else:
        return "NEUTRAL"

def run_linear_regression(df):
    """
    Train linear regression model on clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    
    Returns
    -------
    dict
        Contains keys: model, signal, prediction
    """
    try:
        df_clean = clean_data(df)
        df_features = create_features(df_clean)
        model = train_model(df_features)
        result = predict(df_features, model)
        
        return {
            'model': model,
            'signal': result['signal'],
            'prediction': result['prediction']
        }
    except Exception as e:
        raise RuntimeError(f"Linear regression training failed: {e}")
