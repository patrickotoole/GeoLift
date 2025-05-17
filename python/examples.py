"""Example usage of the Python GeoLift implementation.

This script generates a small synthetic dataset, fits the
``SyntheticControlModel`` and visualises the observed treated
series against the modelled counterfactual using ``matplotlib``.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geolift import SyntheticControlModel


def main() -> None:
    np.random.seed(0)

    time = np.arange(20)
    control_1 = np.random.normal(10, 1, size=20).cumsum()
    control_2 = np.random.normal(10, 1, size=20).cumsum()

    treated = 0.5 * control_1 + 0.5 * control_2
    treated[12:] += 5.0  # introduce lift after period 11

    df = pd.DataFrame({
        "time": time,
        "treated": treated,
        "control_1": control_1,
        "control_2": control_2,
    })

    cutoff = 10
    pre = df[df.time <= cutoff]
    post_controls = df[df.time > cutoff].drop(columns=["treated"])

    model = SyntheticControlModel(pre)
    model.fit(draws=200, tune=200)

    pred = model.predict(post_controls)

    plt.plot(df.time, df.treated, label="treated", color="black")
    plt.plot(post_controls.time, pred, label="counterfactual", linestyle="--", color="red")
    plt.axvline(cutoff, color="gray", linestyle=":")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("outcome")
    plt.title("Synthetic control example")
    plt.show()


if __name__ == "__main__":
    main()
