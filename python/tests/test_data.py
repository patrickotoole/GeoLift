import pandas as pd
from geolift.data import GeoData


def test_geodata_read_lowercase_sorted_time():
    df = pd.DataFrame({
        'date': ['2020-01-02', '2020-01-01', '2020-01-02', '2020-01-01'],
        'location': ['A', 'A', 'B', 'B'],
        'Y': [2, 1, 4, 3],
    })
    geo = GeoData.read(df)
    assert list(geo.data['location']) == ['a', 'a', 'b', 'b']
    assert list(geo.data['time']) == [1, 2, 1, 2]
