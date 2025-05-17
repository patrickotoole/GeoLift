import importlib.util
import unittest

pandas_available = importlib.util.find_spec("pandas") is not None
numpy_available = importlib.util.find_spec("numpy") is not None
pymc_available = importlib.util.find_spec("pymc") is not None

if pandas_available and numpy_available:
    import numpy as np
    import pandas as pd
    from geolift import SyntheticControlModel


class TestSyntheticControlModel(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available and pymc_available, "Required packages are not installed")
    def test_fit_predict(self) -> None:
        df = pd.DataFrame({
            "time": [1, 2, 3],
            "treated": [10.0, 12.0, 13.0],
            "control_1": [9.0, 11.0, 10.0],
            "control_2": [8.0, 12.0, 12.0],
        })

        model = SyntheticControlModel(df)
        model.fit(draws=10, tune=10, seed=42)

        post = pd.DataFrame({
            "time": [4, 5],
            "control_1": [11.0, 10.0],
            "control_2": [13.0, 12.0],
        })
        pred = model.predict(post)
        self.assertEqual(len(pred), 2)


if __name__ == "__main__":
    unittest.main()
