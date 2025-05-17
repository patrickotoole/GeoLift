import unittest

try:
    import pandas as pd
    from geolift.market_selection import prepare_wide_panel, select_test_market
except Exception:  # pragma: no cover - package missing
    pd = None


@unittest.skipIf(pd is None, "pandas not available")
class TestMarketSelection(unittest.TestCase):
    def test_select_market(self):
        data = {
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "market": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "outcome": [1, 2, 3, 1, 1, 1, 1, 1, 1],
        }
        df = pd.DataFrame(data)
        panel = prepare_wide_panel(df, "market", "outcome", "time")
        market = select_test_market(panel, pre_period_end=2)
        self.assertEqual(market, "A")


if __name__ == "__main__":
    unittest.main()
