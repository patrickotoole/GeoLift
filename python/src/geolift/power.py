"""Power calculations for GeoLift using Bayesian regression."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .data import GeoData
from .modeling import BayesianSCM


@dataclass
class PowerResult:
    effect_size: float
    probability_detected: float


def compute_power(
    geo: GeoData,
    treatment_locs: List[str],
    effect_sizes: List[float],
    n_simulations: int = 100,
) -> List[PowerResult]:
    """Compute power curves for several effect sizes.

    This is a simplified version of ``GeoLiftPower`` using a Bayesian
    regression model.
    """
    results: List[PowerResult] = []
    df = geo.data.copy()
    treatments = [loc.lower() for loc in treatment_locs]
    df["is_treatment"] = df["location"].isin(treatments)

    control = df[~df["is_treatment"]]
    treatment = df[df["is_treatment"]]

    model = BayesianSCM(
        control.pivot(index="time", columns="location", values="Y"),
        treatment.groupby("time")["Y"].sum().reset_index(),
    )
    model.fit()

    for es in effect_sizes:
        detected = 0
        for _ in range(n_simulations):
            synthetic = model.predict(
                control.pivot(index="time", columns="location", values="Y")
            )
            observed = treatment.groupby("time")["Y"].sum().values
            lift = observed - synthetic
            if lift.mean() > es:
                detected += 1
        results.append(
            PowerResult(
                effect_size=es,
                probability_detected=detected / n_simulations,
            )
        )
    return results
