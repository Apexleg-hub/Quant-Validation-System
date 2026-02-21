"""
Volatility Prediction using GARCH Model

This module predicts volatility using ARCH/GARCH models on clean data.
"""

from arch import arch_model
import math
import numpy as np
import pandas as pd
from ingestion.data_cleaner import clean_data, create_features


def predict_volatility(df):
    """
    Predict volatility using GARCH model on clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    
    Returns
    -------
    dict
        Contains keys: daily_volatility, monthly_volatility, annual_volatility, 
        forecast, model, params
    """
    try:
        # Clean and feature engineer data
        df_clean = clean_data(df)
        df_features = create_features(df_clean)
        
        # Get close column (handle both cases)
        close_col = 'close' if 'close' in df_features.columns else 'Close'
        if close_col not in df_features.columns:
            raise ValueError(f"Close price column not found in dataframe")
        
        # Calculate returns as percentage
        df_features['Return'] = 100 * (df_features[close_col].pct_change())
        returns = df_features['Return'].dropna()
        
        # Calculate volatilities
        daily_volatility = returns.std()
        monthly_volatility = math.sqrt(21) * daily_volatility
        annual_volatility = math.sqrt(252) * daily_volatility
        
        # Train GARCH model if sufficient data
        if len(returns) > 50:
            garch_model_obj = arch_model(
                returns,
                p=1, q=1,
                mean='constant',
                vol='Garch',
                dist='normal'
            )
            gm_result = garch_model_obj.fit(disp='off')
            gm_forecast = gm_result.forecast(horizon=5)
            
            return {
                'daily_volatility': daily_volatility,
                'monthly_volatility': monthly_volatility,
                'annual_volatility': annual_volatility,
                'forecast': gm_forecast.variance[-1:].values,
                'model': gm_result,
                'params': gm_result.params.to_dict()
            }
        else:
            return {
                'daily_volatility': daily_volatility,
                'monthly_volatility': monthly_volatility,
                'annual_volatility': annual_volatility,
                'forecast': None,
                'model': None,
                'params': None,
                'warning': 'Insufficient data for GARCH model (need >50 observations)'
            }
    except Exception as e:
        raise RuntimeError(f"Volatility prediction failed: {e}")
        