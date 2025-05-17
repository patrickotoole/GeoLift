import unittest

try:
    import pandas as pd
    import numpy as np
    import pymc as pm
    from geolift.synthetic_control import SyntheticControlModel
except Exception:  # pragma: no cover - dependencies missing
    pd = None


@unittest.skipIf(pd is None, "dependencies not available")
class TestSyntheticControl(unittest.TestCase):
    def test_fit_predict(self):
        df = pd.DataFrame({
            "time": [0, 1, 2],
            "treated": [1.0, 2.0, 3.0],
            "control1": [1.0, 1.0, 1.0],
            "control2": [1.0, 1.0, 1.0],
        })
        scm = SyntheticControlModel(df)
        scm.fit(draws=10, tune=10, seed=1)
        pred = scm.predict(df.drop(columns=["treated"]))
        self.assertEqual(len(pred), 3)


if __name__ == "__main__":
    unittest.main()
