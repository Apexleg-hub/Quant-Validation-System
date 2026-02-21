import pandas as pd
from ingestion.data_cleaner import clean_data, create_features, train_model, predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def _pick_col(data: pd.DataFrame, *candidates: str) -> str:
    for name in candidates:
        if name in data.columns:
            return name
    raise KeyError(f"Missing required columns. Tried: {candidates}")


def svm_trade_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight trade filter using SVM signals or EMA as fallback.
    If ML outputs exist, use them; otherwise use an EMA trend proxy.
    """
    out = data.copy()
    close_col = _pick_col(out, "close", "Close")

    if "Signal" not in out.columns:
        out["Signal"] = 0

    if "ml_signal" in out.columns:
        out["Signal"] = out["ml_signal"].replace({0: -1, 1: 1})
    elif "final_signal" in out.columns:
        out["Signal"] = out["final_signal"]
    else:
        out["EMA50"] = out[close_col].ewm(span=50, adjust=False).mean()
        out["EMA200"] = out[close_col].ewm(span=200, adjust=False).mean()
        out.loc[out["EMA50"] > out["EMA200"], "Signal"] = 1
        out.loc[out["EMA50"] < out["EMA200"], "Signal"] = -1

    if "volatility_filter" in out.columns:
        out["Signal"] = out["Signal"] * out["volatility_filter"]

    return out


def run_svm(df):
    """
    Train SVM model on clean data with engineered features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    
    Returns
    -------
    dict
        Contains keys: model, signal, accuracy, report
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
        raise RuntimeError(f"SVM training failed: {e}")

    return model
def predict_svm(model, data: pd.DataFrame, feature_cols: list) -> pd.Series:
    X = data[feature_cols]
    return pd.Series(model.predict(X), index=data.index)
def run_svm(data: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
    cleaned_data = clean_data(data)
    features_data = create_features(cleaned_data)
    model = train_svm(features_data, feature_cols, target_col)
    features_data["ml_signal"] = predict_svm(model, features_data, feature_cols)
    return svm_trade_filter(features_data)

