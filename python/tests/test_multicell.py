import pandas as pd
from geolift.data import GeoData
from geolift.multicell import multicell_market_selection, multicell_power
from geolift.power import PowerResult
from geolift import power as power_module


def _geo():
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=2).tolist() * 4,
        'location': ['A'] * 2 + ['B'] * 2 + ['C'] * 2 + ['D'] * 2,
        'Y': [1, 2] * 4,
    })
    return GeoData.read(df)


def test_multicell_market_selection(monkeypatch):
    geo = _geo()
    monkeypatch.setattr(
        "geolift.multicell.compute_power",
        lambda *args, **kwargs: [PowerResult(0.1, 0.5)],
    )
    result = multicell_market_selection(geo, k=2, top_n=1)
    assert set(result.cells.keys()) == {1, 2}
    total = sum(len(c.locations) for c in result.cells.values())
    assert total == 4


def test_selection_uses_power(monkeypatch):
    geo = _geo()
    calls = {"n": 0}

    def fake_power(*args, **kwargs):
        calls["n"] += 1
        return [PowerResult(0.1, 0.5)]

    monkeypatch.setattr("geolift.multicell.compute_power", fake_power)
    multicell_market_selection(geo, k=2, top_n=1, effect_sizes=[0.1])
    assert calls["n"] > 0


def test_multicell_power(monkeypatch):
    geo = _geo()
    monkeypatch.setattr(
        'geolift.multicell.compute_power',
        lambda g, locs, es, **kwargs: [PowerResult(e, 0.5) for e in es],
    )
    mc = multicell_market_selection(geo, k=2, top_n=1)
    treatment_map = {1: ['A'], 2: ['C']}
    res = multicell_power(geo, mc, treatment_map, [0.1, 0.2])
    assert 1 in res.power and 2 in res.power
    assert len(res.power[1]) == 2
