import pandas as pd
from geolift.data import GeoData
from geolift.market_selection import market_correlations, market_selection


def _make_geo():
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=4).tolist() * 3,
        'location': ['A'] * 4 + ['B'] * 4 + ['C'] * 4,
        'Y': [1, 2, 3, 4] * 3,
    })
    return GeoData.read(df)


def test_market_correlations():
    geo = _make_geo()
    corr = market_correlations(geo.data)
    ab = corr[(corr['var1'] == 'a') & (corr['var2'] == 'b')].iloc[0]
    assert ab.correlation == 1.0


def test_market_selection():
    geo = _make_geo()
    sel = market_selection(geo, top_n=1)
    assert len(sel) == 3
    assert set(sel['var1']) == {'a', 'b', 'c'}

    sel_excl = market_selection(geo, exclude_markets=['b'], top_n=1)
    assert set(sel_excl['var1']) == {'a', 'c'}
