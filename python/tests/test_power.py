import numpy as np
import pandas as pd
from geolift.data import GeoData
from geolift import power as power_module
from geolift.power import compute_power, PowerResult


def test_compute_power(monkeypatch):
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=3).tolist() * 2,
        'location': ['A'] * 3 + ['B'] * 3,
        'Y': [10, 10, 10] * 2,
    })
    geo = GeoData.read(df)

    class DummyModel:
        def __init__(self, control, treatment):
            pass

        def fit(self):
            pass

        def predict(self, control_new):
            return np.zeros(control_new.shape[0])

    monkeypatch.setattr(power_module, 'BayesianSCM', DummyModel)

    results = compute_power(geo, ['A'], effect_sizes=[0.5], n_simulations=5)
    assert isinstance(results[0], PowerResult)
    assert results[0].probability_detected == 1
