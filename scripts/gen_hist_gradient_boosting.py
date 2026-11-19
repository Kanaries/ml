"""Generate fixed-round sklearn references for histogram boosting on numeric data with NaNs."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


def json_rows(matrix):
    # JSON has no NaN literal. The JS test restores these entries to Number.NaN.
    return [[None if np.isnan(value) else float(value) for value in row] for row in matrix]


def main():
    rng = np.random.default_rng(2026)
    X = rng.integers(-90, 91, size=(90, 3)).astype(float) / 16
    X[rng.random(X.shape) < 0.17] = np.nan
    train, test = X[:72], X[72:]
    test[0] = np.nan
    signal = (np.nan_to_num(train[:, 0], nan=-4) + 0.4 * np.nan_to_num(train[:, 1], nan=3)
              + 0.3 * rng.uniform(-1, 1, len(train)))
    targets = {
        'regression': signal.round(4),
        'binary': np.where(signal > 0.25, 11, -3),
        'multiclass': np.array([-3, 4, 11])[np.digitize(signal, [-1.25, 1.3])],
    }
    weights = rng.integers(1, 15, len(train)) / 8
    weights[::13] = 0
    options = dict(max_bins=9, min_samples_leaf=5, max_leaf_nodes=4, max_iter=8,
                   max_depth=3, learning_rate=0.15, l2_regularization=0.3)
    cases = []
    with threadpool_limits(limits=1):
        for task, y in targets.items():
            for weighted in (False, True):
                estimator = HistGradientBoostingRegressor if task == 'regression' else HistGradientBoostingClassifier
                model = estimator(**options, early_stopping=False, random_state=0)
                model.fit(train, y, sample_weight=weights if weighted else None)
                case = dict(task=task, weighted=weighted, prediction=model.predict(test).tolist())
                if task != 'regression':
                    case.update(classes=model.classes_.tolist(), probability=model.predict_proba(test).tolist(),
                                decision=model.decision_function(test).tolist())
                cases.append(case)

    fixture = dict(trainX=json_rows(train), testX=json_rows(test), targets={k: v.tolist() for k, v in targets.items()},
                   weights=weights.tolist(), options=options, cases=cases)
    path = Path(__file__).resolve().parents[1] / 'test_data/hist_gradient_boosting.json'
    path.write_text(json.dumps(fixture, allow_nan=False, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
