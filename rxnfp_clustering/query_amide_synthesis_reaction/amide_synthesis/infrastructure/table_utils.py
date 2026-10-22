"""Shared table operations with stable behavior for absent optional columns."""
import pandas as pd


def column(frame, name, default=''):
    """Return a column or a default Series aligned to the original index."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def nullable_column(frame, name, default=None):
    """Nullable-default adapter for evidence flags; shares the column implementation."""
    return column(frame, name, default)
