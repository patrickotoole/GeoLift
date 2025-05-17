"""Example usage of the GeoLift Python implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geolift import (
    SyntheticControlModel,
    pivot_markets,
    choose_test_market,
)

# Generate synthetic data for illustration
rng = np.random.default_rng(42)
markets = [f"m{i}" for i in range(5)]
periods = pd.date_range("2021-01-01", periods=50, freq="D")

records = []
for m in markets:
    base = rng.normal(scale=0.5, size=len(periods)).cumsum()
    records.extend({"time": t, "market": m, "sales": b + rng.normal(scale=0.1)} for t, b in zip(periods, base))

df_long = pd.DataFrame.from_records(records)

# Pivot into wide format and select test market
wide = pivot_markets(df_long, "market", "time", "sales")
pre_period = slice("2021-01-01", "2021-02-15")
test_market = choose_test_market(wide, pre_period)

print("Selected test market:", test_market)

controls = [c for c in wide.columns if c != test_market]
pre = (
    wide.loc[pre_period, [test_market] + controls]
    .rename(columns={test_market: "treated"})
    .reset_index()
)

scm = SyntheticControlModel(pre)
scm.fit(draws=1000, tune=500, seed=1)
print("Posterior mean weights:\n", scm.weight_means())

post_period = slice("2021-02-16", periods[-1])
post_controls = wide.loc[post_period, controls].reset_index()
counterfactual = scm.predict(post_controls)

# Plot
fig, ax = plt.subplots()
ax.plot(wide.index, wide[test_market], label="Observed")
ax.plot(counterfactual.index, counterfactual, label="Counterfactual")
ax.axvline(pd.to_datetime("2021-02-16"), color="k", ls="--")
ax.legend()
ax.set_ylabel("Sales")
ax.set_title("Observed vs Counterfactual for {}".format(test_market))
plt.show()

