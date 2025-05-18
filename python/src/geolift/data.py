"""Data utilities for the Python port of GeoLift.

This module provides a simplified implementation of the
``GeoDataRead`` functionality from the R package. It accepts a
``pandas`` ``DataFrame`` and ensures the columns used by the library are
present and properly formatted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class GeoData:
    """Container for GeoLift data."""

    data: pd.DataFrame
    date_col: str = "date"
    location_col: str = "location"
    outcome_col: str = "Y"
    covariates: Optional[List[str]] = None

    @classmethod
    def read(
        cls,
        df: pd.DataFrame,
        *,
        date_col: str = "date",
        location_col: str = "location",
        outcome_col: str = "Y",
        covariates: Optional[List[str]] = None,
    ) -> "GeoData":
        """Create a :class:`GeoData` instance from a DataFrame."""
        data = df.rename(
            columns={
                date_col: "date",
                location_col: "location",
                outcome_col: "Y",
            }
        )
        if covariates:
            data = data[["date", "location", "Y", *covariates]]
        else:
            data = data[["date", "location", "Y"]]

        data["location"] = data["location"].str.lower()
        data["date"] = pd.to_datetime(data["date"])

        # Sort for consistency
        data = data.sort_values(["location", "date"]).reset_index(drop=True)
        data["time"] = (
            data.groupby("location")["date"].rank(method="dense").astype(int)
        )
        return cls(data=data, covariates=covariates)
