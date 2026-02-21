import pandas as pd
from processing.data_cleaner import clean_data, create_features, train_model, predict



def _pick_col(data: pd.DataFrame, *candidates: str) -> str:
    for name in candidates:
        if name in data.columns:
            return name
    raise KeyError(f"Missing required columns. Tried: {candidates}")


def svm_trade_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight trade filter.
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
