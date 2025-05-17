# GeoLift Python

This directory hosts a minimal Python implementation of the GeoLift
methodology.  The implementation is intentionally lightweight and
focuses on the creation of synthetic control groups using a Bayesian
approach implemented with [PyMC](https://www.pymc.io/).

The central class `SyntheticControlModel` estimates the synthetic
control weights via Monte Carlo sampling and produces a distribution of
possible counterfactual outcomes.  This allows communicating
uncertainty when reporting lift estimates.

Input data is expected in "long" format with columns identifying the
market, time period and observed value.  The helper function
`pivot_markets` converts this to the wide format required by
`SyntheticControlModel`.  `choose_test_market` can automatically pick a
market to hold out for testing by selecting the one that is least
correlated with the others during the pre-treatment period.

## Example

```python
import pandas as pd
from geolift import (
    SyntheticControlModel,
    pivot_markets,
    choose_test_market,
)

# DataFrame `df_long` should contain columns:
# * `market`
# * `time`
# * `sales` (outcome variable)

# Convert to wide format and automatically choose a test market
df = pivot_markets(df_long, "market", "time", "sales")
test_market = choose_test_market(df, slice("2021-01-01", "2021-02-15"))
controls = [c for c in df.columns if c != test_market]

# Build pre-treatment dataset expected by SyntheticControlModel
pre = (
    df.loc["2021-01-01":"2021-02-15", [test_market] + controls]
    .rename(columns={test_market: "treated"})
    .reset_index()
)
post_controls = df.loc["2021-02-16":, controls].reset_index()

scm = SyntheticControlModel(pre)
scm.fit(draws=2000, tune=1000)

counterfactual = scm.predict(post_controls)
# Posterior mean weights for the control markets
weights = scm.weight_means()
```

The `weight_means` method returns the posterior mean contribution of each control market. Plotting the observed test market against the predicted counterfactual provides a visual interpretation of lift.

