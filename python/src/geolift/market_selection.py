"""Market selection routines."""
from __future__ import annotations

from typing import List

import pandas as pd

from .data import GeoData


def market_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Compute within market correlations.

    Parameters
    ----------
    data: DataFrame
        Data in long format with columns ``location``, ``time`` and ``Y``.
    Returns
    -------
    DataFrame
        Pairwise correlations between markets.
    """
    pivot = data.pivot_table(index="time", columns="location", values="Y")
    corr = pivot.corr()
    corr["var1"] = corr.index
    long = corr.melt(id_vars="var1", var_name="var2", value_name="correlation")
    return long


def market_selection(
    geo: GeoData,
    exclude_markets: List[str] | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top correlated markets for each location."""
    data = geo.data.copy()
    if exclude_markets:
        exclusions = [m.lower() for m in exclude_markets]
        data = data[~data["location"].isin(exclusions)]

    corr = market_correlations(data)
    # For each var1 sort by correlation and take top_n
    best = (
        corr.sort_values(["var1", "correlation"], ascending=[True, False])
        .groupby("var1")
        .head(top_n)
    )
    return best.reset_index(drop=True)
