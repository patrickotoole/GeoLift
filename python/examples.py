"""Demonstration of the GeoLift Python implementation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geolift import SyntheticControlModel, prepare_wide_panel, select_test_market


# Generate a synthetic panel of markets
np.random.seed(1)
T = 50
markets = ["A", "B", "C", "D"]
records = []
for m in markets:
    trend = np.linspace(0, 5, T)
    noise = np.random.normal(scale=0.5, size=T)
    outcome = trend + noise
    if m == "A":
        outcome += np.where(np.arange(T) > 30, 3.0, 0)  # effect after time 30
    for t, y in enumerate(outcome):
        records.append({"time": t, "market": m, "outcome": y})

df_long = pd.DataFrame(records)

# Prepare wide panel and automatically choose a test market
panel = prepare_wide_panel(df_long, "market", "outcome", "time")
test_market = select_test_market(panel, pre_period_end=30)

pre = panel[panel.time <= 30].rename(columns={test_market: "treated"})
post_controls = panel[panel.time > 30].drop(columns=[test_market])

# Fit synthetic control model and predict counterfactual
scm = SyntheticControlModel(pre)
scm.fit(draws=1000, tune=500, seed=42)

counterfactual = scm.predict(post_controls)
actual = panel.loc[panel.time > 30, test_market].reset_index(drop=True)

# Plot observed vs. counterfactual
plt.plot(post_controls.time.values, actual, label="Observed")
plt.plot(post_controls.time.values, counterfactual, label="Counterfactual")
plt.xlabel("time")
plt.ylabel("outcome")
plt.title(f"Test market: {test_market}")
plt.legend()
plt.tight_layout()
plt.show()
