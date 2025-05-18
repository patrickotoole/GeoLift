"""Power calculations for GeoLift using Bayesian regression."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from scipy import stats

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
    *,
    duration: Optional[int] = None,
    alpha: float = 0.1,
    side_of_test: str = "two_sided",
    n_simulations: int = 1,
) -> List[PowerResult]:
    """Compute power curves for several effect sizes.

    Parameters
    ----------
    geo:
        Input data container.
    treatment_locs:
        Locations to be considered as treatment.
    effect_sizes:
        List of proportional lifts to test.
    duration:
        Number of time periods for the simulated test. If ``None`` the entire
        time series is used.
    alpha:
        Significance level for the statistical test.
    side_of_test:
        Either ``"one_sided"`` or ``"two_sided"`` indicating the type of test.
    n_simulations:
        Number of repeated simulations. The simplified implementation uses the
        same data for each simulation so this primarily mirrors the R API.
    """
    if side_of_test not in {"one_sided", "two_sided"}:
        raise ValueError("side_of_test must be 'one_sided' or 'two_sided'")

    df = geo.data.copy()
    treatments = [loc.lower() for loc in treatment_locs]
    df["is_treatment"] = df["location"].isin(treatments)

    times = sorted(df["time"].unique())
    if duration is None or duration > len(times):
        duration = len(times)
    start_idx = len(times) - duration
    start_time = times[start_idx]

    results: List[PowerResult] = []

    for es in effect_sizes:
        df_es = df.copy()
        df_es.loc[df_es["is_treatment"] & (df_es["time"] >= start_time), "Y"] *= (
            1 + es
        )

        pre = df_es[df_es["time"] < start_time]
        control_mat = pre[~pre["is_treatment"]].pivot(
            index="time", columns="location", values="Y"
        )
        treatment_series = (
            pre[pre["is_treatment"]].groupby("time")["Y"].sum().reset_index()
        )

        model = BayesianSCM(control_mat, treatment_series)
        model.fit()

        control_all = df_es[~df_es["is_treatment"]].pivot(
            index="time", columns="location", values="Y"
        )
        synthetic = model.predict(control_all)
        observed = df_es[df_es["is_treatment"]].groupby("time")["Y"].sum().values
        lift = observed - synthetic
        post_lift = lift[start_idx:]

        if side_of_test == "one_sided":
            alt = "greater" if es > 0 else "less"
            _, pvalue = stats.ttest_1samp(post_lift, 0.0, alternative=alt)
        else:
            _, pvalue = stats.ttest_1samp(post_lift, 0.0)

        probability = float(pvalue < alpha)
        results.append(PowerResult(effect_size=es, probability_detected=probability))

    return results
