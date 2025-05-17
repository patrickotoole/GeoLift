import unittest
import importlib

pd_spec = importlib.util.find_spec("pandas")
np_spec = importlib.util.find_spec("numpy")

@unittest.skipUnless(pd_spec and np_spec, "requires pandas and numpy")
class TestMarketSelection(unittest.TestCase):
    def test_choose_market(self):
        import numpy as np
        import pandas as pd
        from geolift import pivot_markets, choose_test_market

        rng = np.random.default_rng(1)
        periods = range(5)
        markets = ["a", "b", "c"]
        records = []
        for m in markets:
            base = rng.normal(scale=0.1, size=len(periods)).cumsum()
            for t, val in zip(periods, base + rng.normal(scale=0.01, size=len(periods))):
                records.append({"time": t, "market": m, "y": val})
        df = pd.DataFrame(records)
        wide = pivot_markets(df, "market", "time", "y")
        m = choose_test_market(wide, slice(0, 3))
        self.assertIn(m, markets)

