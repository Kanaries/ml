"""Generate deterministic CPU reference updates from the pinned Faiss release."""
import json
from pathlib import Path

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means.base import _resolve_batch_law, _resolve_recipe

faiss.omp_set_num_threads(1)


class FixedDonorDraws:
    def random_sample(self, size):
        return np.full(size, 0.4)


def orders(n, epochs, seed):
    # Match the library's seeded shuffle, keeping the numerical oracle independent.
    order = list(range(n))
    for _ in range(epochs):
        for i in range(n):
            seed = seed * 16807 % 2147483647
            j = i + int((seed - 1) / 2147483646 * (n - i))
            order[i], order[j] = order[j], order[i]
        yield order.copy()


def trace(threshold):
    x = np.array([[-4, 2, 0], [-3, 2, 1], [-3, 3, 0], [-1, 2, 1],
                  [2, 1, 2], [4, 3, 0], [5, 1, 1], [3, 2, 0], [2, 1, 1]], dtype='float32')
    init = np.array([x[0], x[4], [80, 90, 70]], dtype='float32')
    model = NumpyEmeans(3, 3, nepochs=2, batch_size=2, unused_threshold=threshold)
    model.begin(len(x), init)
    model._split_rng = FixedDonorDraws()
    batches = []
    for epoch, order in enumerate(orders(len(x), 2, 17)):
        for start in range(0, len(x), 2):
            ids = order[start:start + 2]
            model.update(model.accumulate(x[ids]))
            batches.append(dict(ids=ids, closing=epoch == 1, centers=model._centroids.tolist(),
                                mass=model._usage.tolist()))
        model.end_epoch()
    final = model.finalize()
    dist, labels = model.assign(x)
    return dict(x=x.tolist(), init=init.tolist(), threshold=threshold, batches=batches,
                splits=model.n_splits, centers=final.tolist(), labels=labels.tolist(),
                inertia=float(dist.sum(dtype='float64')))


def recipe(n, d, k, epochs, batch=None):
    law, _ = _resolve_batch_law('auto', d)
    size, alpha = _resolve_recipe(n, k, epochs, law, batch, None)
    return dict(n=n, d=d, k=k, epochs=epochs, batch=batch, size=size, alpha=alpha)


data = dict(traces=[trace(0), trace(0.1)], recipes=[
    recipe(1, 1, 1, 1), recipe(9, 3, 3, 2, 2), recipe(1000, 16, 8, 5),
    recipe(1000, 17, 8, 5), recipe(11, 16, 5, 90), recipe(11, 17, 5, 90),
    recipe(31, 8, 2, 5, 31), recipe(100_000, 128, 256, 3)])
path = Path(__file__).resolve().parents[1] / 'test_data' / 'e_means.json'
path.write_text(json.dumps(data, indent=2) + '\n')
