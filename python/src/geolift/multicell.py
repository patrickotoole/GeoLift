"""Multi-cell experimentation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable

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
    effect_sizes: Iterable[float] | None = None,
    duration: int | None = None,
    alpha: float = 0.1,
    side_of_test: str = "two_sided",
) -> MultiCellResult:
    """Create ``k`` partitions of locations for testing.

    Each cell is evaluated using :func:`compute_power` so that the most
    promising markets can be surfaced.
    """
    if effect_sizes is None:
        effect_sizes = [0.1]
    unique_locs = geo.data["location"].unique()
    splits = [unique_locs[i::k] for i in range(k)]
    cells: Dict[int, MultiCellMarket] = {}
    for idx, locs in enumerate(splits, start=1):
        sub_geo = GeoData(geo.data[geo.data["location"].isin(locs)].copy())

        candidates = []
        for loc in locs:
            power = compute_power(
                sub_geo,
                [loc],
                list(effect_sizes),
                duration=duration,
                alpha=alpha,
                side_of_test=side_of_test,
            )
            avg_power = sum(p.probability_detected for p in power) / len(power)
            candidates.append({"location": loc, "avg_power": avg_power})

        sel = (
            pd.DataFrame(candidates)
            .sort_values("avg_power", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

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
    *,
    duration: int | None = None,
    alpha: float = 0.1,
    side_of_test: str = "two_sided",
) -> MultiCellResult:
    """Estimate power curves for each cell.

    Parameters are forwarded to :func:`compute_power` for each cell.
    """
    power_results: Dict[int, List[PowerResult]] = {}
    for cell_id, locations in treatment_map.items():
        power_results[cell_id] = compute_power(
            geo,
            locations,
            effect_sizes,
            duration=duration,
            alpha=alpha,
            side_of_test=side_of_test,
        )
    multicell.power = power_results
    return multicell
