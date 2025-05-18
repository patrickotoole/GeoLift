"""Bayesian models used in GeoLift."""
from __future__ import annotations

import pymc as pm
import numpy as np
import pandas as pd
import xarray as xr


class BayesianSCM:
    """Simple Bayesian synthetic control using PyMC."""

    def __init__(self, control: pd.DataFrame, treatment: pd.DataFrame):
        self.control = control
        self.treatment = treatment
        self.trace: xr.InferenceData | None = None

    def fit(self) -> xr.InferenceData:
        """Fit a regression of treatment on controls."""
        y = self.treatment["Y"].values
        X = self.control.values
        n, k = X.shape
        with pm.Model():
            beta = pm.Normal("beta", 0, 1, shape=k)
            sigma = pm.Exponential("sigma", 1)
            mu = pm.math.dot(X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
            trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
        self.trace = trace
        return trace

    def predict(self, control_new: pd.DataFrame) -> np.ndarray:
        if self.trace is None:
            raise RuntimeError("Model is not fitted")
        beta = self.trace.posterior["beta"].mean(dim=("chain", "draw"))
        mu = np.dot(control_new.values, beta)
        return mu
