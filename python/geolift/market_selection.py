"""Helpers for selecting test markets from a panel dataset."""

from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np


def prepare_wide_panel(
    df: pd.DataFrame,
    market_col: str,
    outcome_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Pivot a long-form DataFrame into wide format by market."""
    wide = df.pivot_table(index=time_col, columns=market_col, values=outcome_col)
    wide = wide.reset_index().sort_values(time_col).reset_index(drop=True)
    wide.columns.name = None
    return wide


def select_test_market(df_wide: pd.DataFrame, pre_period_end: int | float) -> str:
    """Select a test market with minimal correlation to the others."""
    market_cols = [c for c in df_wide.columns if c != "time"]
    pre = df_wide[df_wide["time"] <= pre_period_end]

    best_market = None
    lowest_corr = None
    for m in market_cols:
        others = [c for c in market_cols if c != m]
        if not others:
            continue
        corr = pre[m].corr(pre[others].mean(axis=1))
        if lowest_corr is None or corr < lowest_corr:
            lowest_corr = corr
            best_market = m
    if best_market is None:
        raise ValueError("Unable to select test market")
    return best_market
