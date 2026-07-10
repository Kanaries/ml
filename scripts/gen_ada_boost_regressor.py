from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import json, os

# Platform-stable data: rng.uniform is pure bit arithmetic (no libm) and the
# polynomial target uses only IEEE-exact ops, so X/y are byte-identical on
# every platform. AdaBoost.R2's weighted resampling amplifies any ULP
# difference into different trees, which is why randn/make_regression-based
# data drifted across OSes in the fixture-drift job.
rng = np.random.RandomState(0)
X = rng.uniform(-3, 3, size=(60, 4))
y = 12.0 * X[:, 0] + 4.0 * X[:, 1] ** 2 - 6.0 * X[:, 2] + rng.uniform(-0.1, 0.1, size=60)

regr = AdaBoostRegressor(random_state=0, n_estimators=50, estimator=DecisionTreeRegressor(max_depth=3))
regr.fit(X, y)
X_test = rng.uniform(-3, 3, size=(10, 4))
pred = regr.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/ada_boost_regressor.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
