"""Utilities for selecting test markets for synthetic control modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_markets(df: pd.DataFrame, market_col: str, time_col: str, value_col: str) -> pd.DataFrame:
    """Pivot long-form market data into wide format.

    Parameters
    ----------
    df: pd.DataFrame
        Input data containing one row per market/time combination.
    market_col: str
        Column identifying each market.
    time_col: str
        Column identifying the time period.
    value_col: str
        Column containing outcome values for each market and time.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time with one column per market.
    """
    wide = df.pivot(index=time_col, columns=market_col, values=value_col)
    wide = wide.sort_index()
    return wide


def choose_test_market(wide: pd.DataFrame, pre_period: slice) -> str:
    """Select a test market with the lowest correlation to others."""
    pre = wide.loc[pre_period]
    corr = pre.corr()
    # Ignore self-correlation by filling diagonal with NaN
    np.fill_diagonal(corr.values, np.nan)
    mean_corr = corr.mean()
    return mean_corr.idxmin()
