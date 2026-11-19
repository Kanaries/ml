"""Freeze exact GPR and Laplace GPC results for fixed sklearn kernels."""
import json
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


X = np.array([[-2, 0], [-1.6, .3], [-.9, 1], [-.3, -.2], [.3, .5], [.8, -.7], [1.4, .1], [2, .8]])
query = np.array([[-2, 0], [-.8, .6], [.9, .4], [3, 2]])
target = np.array([1.1, .8, 1.6, 3.1, 4.7, 3.7, 4, 2.9])


def kernel(scale, variance):
    return ConstantKernel(variance, constant_value_bounds="fixed") * RBF(scale, length_scale_bounds="fixed")


def regression(scale, variance, alpha, normalize):
    regressor = GaussianProcessRegressor(
        kernel=kernel(scale, variance), alpha=np.array(alpha) if isinstance(alpha, list) else alpha,
        optimizer=None, normalize_y=normalize
    ).fit(X, target)
    mean, std = regressor.predict(query, return_std=True)
    _, cov = regressor.predict(query, return_cov=True)
    return {
        "options": {"lengthScale": scale, "signalVariance": variance, "alpha": alpha, "normalizeY": normalize},
        "mean": mean.tolist(), "std": std.tolist(), "cov": cov.tolist(), "lml": regressor.log_marginal_likelihood(),
    }


def classification(scale, variance, labels):
    classifier = GaussianProcessClassifier(kernel=kernel(scale, variance), optimizer=None).fit(X, labels)
    return {
        "options": {"lengthScale": scale, "signalVariance": variance}, "y": labels,
        "classes": classifier.classes_.tolist(), "predicted": classifier.predict(query).tolist(),
        "probabilities": classifier.predict_proba(query).tolist(),
    }


fixture = {
    "X": X.tolist(), "query": query.tolist(), "target": target.tolist(),
    "regressions": [
        regression(.8, 1.5, .04, False),
        regression([1.2, .7], 2, [.01, .04, .02, .1, .03, .05, .08, .02], True),
    ],
    "classifiers": [
        classification(.8, 1.5, [-5, -5, -5, 8, 8, -5, 8, 8]),
        classification([1.2, .7], 2, [-5, -5, 3, 3, 11, 3, 11, 11]),
    ],
}
path = Path(__file__).resolve().parents[1] / "test_data" / "gaussian_process.json"
path.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
