import numpy as np

from abcd import ABCD


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    datastream_1 = np.concatenate([
        rng.uniform(0, 0.5, size=2000),
        rng.uniform(0.5, 1, size=2000),
        rng.uniform(0.25, 0.75, size=2000)
    ])

    datastream_2 = np.concatenate([
        rng.uniform(0, 0.5, size=(2000, 1)),
        rng.uniform(0.5, 1, size=(2000, 1)),
        rng.uniform(0.25, 0.75, size=(2000, 1))
    ])

    datastream_2 = np.concatenate([
        rng.uniform(0, 0.5, size=(2000, 1)),
        rng.uniform(0.5, 1, size=(2000, 1)),
        rng.uniform(0.25, 0.75, size=(2000, 1))
    ])

    datastream_3 = np.concatenate([
        rng.uniform(0, 0.5, size=(2000, 10)),
        rng.uniform(0.5, 1, size=(2000, 10)),
        rng.uniform(0.25, 0.75, size=(2000, 10))
    ])

    datastream_4 = [
        {f"feature {i}": f for (i, f) in enumerate(x)}
        for x in datastream_2
    ]

    datastream_5 = [
        {f"feature {i}": f for (i, f) in enumerate(x)}
        for x in datastream_3
    ]

    datastream_6_nochange = rng.uniform(size=(3 * 2000, 5), low=0.3, high=.7)
    datastream_6_change = np.concatenate([
        rng.uniform(0, 0.5, size=(2000, 5)),
        rng.uniform(0.5, 1, size=(2000, 5)),
        rng.uniform(0.25, 0.75, size=(2000, 5))
    ])
    datastream_6 = np.concatenate([datastream_6_nochange, datastream_6_change], axis=1)

    streams = [
        datastream_1,
        datastream_2,
        datastream_3,
        datastream_4,
        datastream_5,
        datastream_6
    ]

    for stream_index, stream in enumerate(streams):
        print(f"*** Stream {stream_index} ***")
        abcd = ABCD()
        for index, x in enumerate(stream):
            abcd.update(x)
            if abcd.drift_detected:
                change_subspace_indices = [i + 1 for i in range(len(abcd.drift_dimensions)) if abcd.drift_dimensions[i] < abcd._subspace_threshold]
                print(f"Change detected at index {index}")
                print(f"Change subspace is {change_subspace_indices}")
                print(f"Change severity is {abcd.severity}")
        print("\n")
