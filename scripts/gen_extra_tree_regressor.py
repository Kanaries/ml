from sklearn.tree import ExtraTreeRegressor
import numpy as np, json, os

# Platform-stable data (uniform + polynomial target, no libm): ExtraTree's
# random split thresholds are drawn from per-node [min, max] ranges, so any
# ULP difference in X flips thresholds and rewrites entire subtrees. See
# gen_ada_boost_regressor.py for the full rationale.
rng = np.random.RandomState(0)
X = rng.uniform(-3, 3, size=(50, 3))
y = 10.0 * X[:, 0] + 5.0 * X[:, 1] ** 2 - 3.0 * X[:, 2] + rng.uniform(-0.1, 0.1, size=50)

reg = ExtraTreeRegressor(random_state=0)
reg.fit(X, y)
X_test = rng.uniform(-3, 3, size=(10, 3))
pred = reg.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/extra_tree_regressor.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
