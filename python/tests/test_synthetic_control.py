import unittest

import importlib

pd_spec = importlib.util.find_spec("pandas")
np_spec = importlib.util.find_spec("numpy")
pymc_spec = importlib.util.find_spec("pymc")

@unittest.skipUnless(pd_spec and np_spec and pymc_spec, "requires pandas, numpy, pymc")
class TestSyntheticControl(unittest.TestCase):
    def test_fit_predict(self):
        import numpy as np
        import pandas as pd
        from geolift import SyntheticControlModel

        rng = np.random.default_rng(0)
        pre = pd.DataFrame({
            "time": [0, 1, 2],
            "treated": [1.0, 2.0, 3.0],
            "c1": rng.normal(size=3),
            "c2": rng.normal(size=3),
        })
        model = SyntheticControlModel(pre)
        model.fit(draws=100, tune=50, seed=1)
        post = pd.DataFrame({
            "time": [3, 4],
            "c1": rng.normal(size=2),
            "c2": rng.normal(size=2),
        })
        pred = model.predict(post)
        self.assertEqual(len(pred), 2)
        w = model.weight_means()
        self.assertAlmostEqual(w.sum(), 1.0, places=2)

