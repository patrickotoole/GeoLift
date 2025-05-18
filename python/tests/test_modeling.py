import pandas as pd
import pymc as pm
from geolift.modeling import BayesianSCM


# Use a lightweight sampler to keep tests fast

def test_bayesian_scm_fit_predict(monkeypatch):
    control = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
    treatment = pd.DataFrame({'Y': [3, 4, 5]})
    model = BayesianSCM(control, treatment)

    orig_sample = pm.sample

    def fast_sample(*args, **kwargs):
        return orig_sample(draws=10, tune=10, chains=1, progressbar=False)

    monkeypatch.setattr(pm, 'sample', fast_sample)
    trace = model.fit()
    assert 'beta' in trace.posterior
    preds = model.predict(control)
    assert len(preds) == len(control)
