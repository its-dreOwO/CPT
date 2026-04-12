import pandas as pd
import numpy as np


def pct_change(old_val: float, new_val: float) -> float:
    """Calculates percentage change between two values."""
    if old_val == 0:
        return 0.0 if new_val == 0 else np.inf * (1 if new_val > 0 else -1)
    return (new_val - old_val) / abs(old_val)


def normalize_price(series: pd.Series) -> pd.Series:
    """
    Min-max scales a pandas Series of prices.
    Returns values bounded between 0 and 1.
    """
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Computes log returns for model input based on a price column.
    """
    # Use log(P_t / P_{t-1})
    return np.log(df[price_col] / df[price_col].shift(1))
