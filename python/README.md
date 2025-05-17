# GeoLift Python

This directory hosts a lightweight Python implementation of the GeoLift
methodology.  The code focuses on building synthetic control groups and
estimating uncertainty using [PyMC](https://www.pymc.io/).

The key pieces are:

* `SyntheticControlModel` &ndash; fits a Bayesian model to pre-treatment data
  and produces a posterior distribution of counterfactual outcomes.
* `prepare_wide_panel` and `select_test_market` &ndash; utilities for working
  with a long form dataset containing many markets.  They pivot the data to
  wide format and automatically choose a test market by finding the market with
  the lowest correlation to the rest during the pre period.

### Input Data

Input should be provided in a **long** format `DataFrame` with columns
`time`, `market`, and `outcome`.  Each row represents the observed outcome for
one market at a point in time.  The helper `prepare_wide_panel` converts this
into the **wide** format expected by `SyntheticControlModel`.

### Running the example

An example script `examples.py` demonstrates how to:

1. Generate a synthetic panel of multiple markets.
2. Automatically select a test market.
3. Fit the model using PyMC.
4. Visualize observed vs. counterfactual outcomes with `matplotlib`.

Run it with:

```bash
python examples.py
```

### Interpreting results

`examples.py` will display a plot showing the observed trajectory of the test
market after the intervention compared to the mean predicted counterfactual.
The gap between these lines represents the estimated lift, while the variation
in predictions (accessible through the posterior samples) conveys the
uncertainty around that lift estimate.
