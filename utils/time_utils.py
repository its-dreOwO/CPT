import datetime
import pandas as pd
from typing import Union


def now_utc() -> datetime.datetime:
    """Returns the current aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)


def to_utc(dt: Union[str, datetime.datetime, pd.Timestamp]) -> datetime.datetime:
    """Converts a string, naive datetime, or aware datetime into an aware UTC datetime."""
    if isinstance(dt, str):
        # Try basic fromisoformat first, else fallback to pd.to_datetime
        try:
            dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            dt = pd.to_datetime(dt).to_pydatetime()

    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)

    return dt.astimezone(datetime.timezone.utc)


def resample_hourly(df: pd.DataFrame, datetime_col: str = None) -> pd.DataFrame:
    """
    Resamples a dataframe to 1h UTC DatetimeIndex.
    Assumes the index is a datetime index if datetime_col is Not provided.
    """
    df = df.copy()
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
        df.set_index(datetime_col, inplace=True)
    else:
        # Ensure the current index is datetime UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    # Forward fill or aggregate depending on what makes sense
    # For now, resample and forward fill missing hours
    resampled = df.resample("1h").ffill()
    return resampled
