import numpy as np

from abcd import ABCD


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    univariate_datastream = np.concatenate([
        rng.uniform(0, 0.5, size=1000),
        rng.uniform(0.5, 1, size=1000)
    ])

    abcd = ABCD()

    for index, x in enumerate(univariate_datastream):
        abcd.update(x)
        if abcd.drift_detected:
            print(f"Change detected at index {index}")
