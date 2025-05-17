# GeoLift Python

This directory contains a lightweight Python port of the original GeoLift R package.  The goal is to build synthetic control groups using a Bayesian approach implemented with [PyMC](https://www.pymc.io/).  The implementation is intentionally minimal but showcases how Monte Carlo sampling can be used to obtain a distribution over counterfactual outcomes.

## Data format

Input data should be provided as a `pandas.DataFrame` with one row per time period.  At a minimum the following columns are expected:

- `time`: ordered timestamps or integer period identifiers
- `treated`: observed outcome of the treated unit during the pre-treatment period
- one column for each potential control unit (for example `control_1`, `control_2`, ...)

Pre-treatment observations for the treated unit are used together with the control units to infer a set of weights that best reproduce the treated series.  Post-treatment periods only require the control columns when generating the counterfactual prediction.

## Interpreting results

`SyntheticControlModel.fit` performs MCMC sampling to estimate the control weights.  When calling `predict` on post-treatment control data the model returns the mean predicted outcome for the treated unit.  The difference between the actual treated series and this prediction represents the estimated lift.  Because a full posterior distribution of the weights is stored, you can also inspect the uncertainty of the prediction (e.g. using credible intervals).

See `examples.py` for a small runnable demonstration that also plots the observed and predicted series using Matplotlib.

## Quick usage

```python
import pandas as pd
from geolift import SyntheticControlModel

# df is a DataFrame as described above
pre = df[df.time <= cutoff]
post_controls = df[df.time > cutoff].drop(columns=["treated"])

scm = SyntheticControlModel(pre)
scm.fit(draws=2000, tune=1000)

counterfactual = scm.predict(post_controls)
```

Run `python examples.py` from this directory to see a full example with plots.
