from sklearn.linear_model import LogisticRegression
from ingestion.data_cleaner import clean_data, create_features, train_model, predict

def run_logistic_regression_v2(df):
    """
    Alternative logistic regression implementation using data_cleaner pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    
    Returns
    -------
    dict
        Contains keys: model, signal, prob_up, prediction
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
        raise RuntimeError(f"Logistic regression v2 training failed: {e}")