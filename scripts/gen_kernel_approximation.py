"""Freeze sklearn's fitted kernel feature maps on independent fit and query samples."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.kernel_approximation import Nystroem, RBFSampler

X = np.array([[1, 0, -1], [0, 2, 1], [-1, 0.5, 2], [2, 1, 0], [0, -1, 3]])
query = np.array([[0, 0, 0], [-0.25, 0, 1], [1.5, 1, -0.5]])
seed = 17
fixture = {"X": X.tolist(), "query": query.tolist(), "seed": seed}
samplers = []
for gamma in [0.75, "scale"]:
    sampler = RBFSampler(gamma=gamma, n_components=5, random_state=seed)
    sampler.fit(csr_matrix(X))
    samplers.append({
        "gamma": gamma,
        "weights": sampler.random_weights_.tolist(),
        "offset": sampler.random_offset_.tolist(),
        "projection": sampler.transform(csr_matrix(query)).tolist(),
    })
fixture["rbfSampler"] = samplers

maps = []
for kernel, params in [
    ("rbf", {"gamma": 0.75}),
    ("poly", {"gamma": 0.5, "degree": 2, "coef0": 0.25}),
    ("cosine", {}),
    ("linear", {}),
]:
    transformer = Nystroem(kernel=kernel, n_components=3, random_state=seed, **params).fit(X)
    maps.append({
        "kernel": kernel,
        "params": params,
        "indices": transformer.component_indices_.tolist(),
        "projection": transformer.transform(query).tolist(),
    })
fixture["nystroem"] = maps

output = Path(__file__).resolve().parents[1] / "test_data" / "kernel_approximation.json"
output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
