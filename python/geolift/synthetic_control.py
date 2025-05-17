"""Synthetic control utilities implemented using PyMC.

This module provides a simple interface for building a synthetic
control model using Bayesian inference. The implementation mirrors the
R library but leverages PyMC to obtain a posterior distribution over
the synthetic control weights and resulting predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class SyntheticControlModel:
    """Bayesian synthetic control model."""

    pre_treatment: pd.DataFrame
    trace: Optional[pm.backends.base.MultiTrace] = None

    def fit(self, draws: int = 2000, tune: int = 1000, seed: Optional[int] = None) -> None:
        """Fit the synthetic control model using MCMC."""
        df = self.pre_treatment.sort_values("time").reset_index(drop=True)
        controls = df.drop(columns=["time", "treated"]).values
        treated = df["treated"].values

        n_controls = controls.shape[1]

        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.ones(n_controls))
            mu = pm.Deterministic("mu", pm.math.dot(controls, w))
            sigma = pm.Exponential("sigma", 1.0)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=treated)

            self.trace = pm.sample(draws=draws, tune=tune, random_seed=seed, progressbar=False)

    def weight_means(self) -> pd.Series:
        """Return posterior mean weights after fitting."""
        if self.trace is None:
            raise RuntimeError("Model must be fit before accessing weights")
        w_samples = self.trace.posterior["w"].stack(draws=("chain", "draw"))
        means = w_samples.mean(dim="draws").values
        control_cols = [c for c in self.pre_treatment.columns if c not in {"time", "treated"}]
        return pd.Series(means, index=control_cols)

    def predict(self, post_controls: pd.DataFrame) -> pd.Series:
        """Predict counterfactual outcomes for the treated unit."""
        if self.trace is None:
            raise RuntimeError("Model must be fit before prediction")

        controls = post_controls.drop(columns=["time"], errors="ignore").values
        w_samples = self.trace.posterior["w"].stack(draws=("chain", "draw")).values

        preds = controls @ w_samples
        pred_mean = preds.mean(axis=1)
        return pd.Series(pred_mean, index=post_controls.index)
