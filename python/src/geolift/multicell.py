"""Multi-cell experimentation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .data import GeoData
from .market_selection import market_selection
from .power import compute_power, PowerResult


@dataclass
class MultiCellMarket:
    cell_id: int
    locations: List[str]
    selection: pd.DataFrame


@dataclass
class MultiCellResult:
    cells: Dict[int, MultiCellMarket]
    power: Dict[int, List[PowerResult]] | None = None


def multicell_market_selection(
    geo: GeoData,
    k: int,
    *,
    top_n: int = 5,
) -> MultiCellResult:
    """Create ``k`` partitions of locations for testing."""
    unique_locs = geo.data["location"].unique()
    splits = [unique_locs[i::k] for i in range(k)]
    cells: Dict[int, MultiCellMarket] = {}
    for idx, locs in enumerate(splits, start=1):
        sub_geo = GeoData(geo.data[geo.data["location"].isin(locs)].copy())
        sel = market_selection(sub_geo, top_n=top_n)
        cells[idx] = MultiCellMarket(
            cell_id=idx,
            locations=list(locs),
            selection=sel,
        )
    return MultiCellResult(cells=cells)


def multicell_power(
    geo: GeoData,
    multicell: MultiCellResult,
    treatment_map: Dict[int, List[str]],
    effect_sizes: List[float],
) -> MultiCellResult:
    """Estimate power curves for each cell."""
    power_results: Dict[int, List[PowerResult]] = {}
    for cell_id, locations in treatment_map.items():
        power_results[cell_id] = compute_power(geo, locations, effect_sizes)
    multicell.power = power_results
    return multicell
