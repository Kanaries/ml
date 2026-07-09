import json, os, sys

try:
    from xgboost import XGBRegressor
except ImportError:
    print('WARNING: xgboost not installed, skipping xgboost_regressor fixture '
          '(the committed fixture stays in place)', file=sys.stderr)
    sys.exit(0)

import numpy as np

rng = np.random.RandomState(0)
X = rng.uniform(-3, 3, size=(120, 2))
y = 4 * X[:, 0] ** 2 - 2 * X[:, 0] + 3 * X[:, 1] + 1
trainX = X[:90]
trainY = y[:90]
testX = X[90:]

reg = XGBRegressor(
    n_estimators=50,
    learning_rate=0.3,
    max_depth=6,
    reg_lambda=1,
    tree_method='exact',
    base_score=0.5,
    random_state=0,
)
reg.fit(trainX, trainY)
pred = reg.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/xgboost_regressor.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': [float(v) for v in pred]
    }, f)
