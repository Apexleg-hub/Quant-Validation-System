import pandas as pd


def _pick_col(df: pd.DataFrame, *candidates: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing required columns. Tried: {candidates}")


def apply_atr_risk(df: pd.DataFrame, atr_window: int = 14, filter_window: int = 50) -> pd.DataFrame:
    """
    Add ATR-based risk columns and a volatility filter.
    Works with either OHLC uppercase or lowercase column names.
    """
    out = df.copy()

    high_col = _pick_col(out, "high", "High")
    low_col = _pick_col(out, "low", "Low")
    close_col = _pick_col(out, "close", "Close")

    out["H-L"] = out[high_col] - out[low_col]
    out["H-C"] = (out[high_col] - out[close_col].shift(1)).abs()
    out["L-C"] = (out[low_col] - out[close_col].shift(1)).abs()
    out["TR"] = out[["H-L", "H-C", "L-C"]].max(axis=1)
    out["ATR"] = out["TR"].rolling(atr_window, min_periods=atr_window).mean()

    out["ATR_mean"] = out["ATR"].rolling(filter_window, min_periods=filter_window).mean()
    out["volatility_filter"] = (out["ATR"] > out["ATR_mean"]).astype(int)

    return out
