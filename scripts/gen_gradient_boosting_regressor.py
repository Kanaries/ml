from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import json, os

rng = np.random.RandomState(0)
X = rng.uniform(-3, 3, size=(120, 2))
y = 4 * X[:, 0] ** 2 - 2 * X[:, 0] + 3 * X[:, 1] + 1
trainX = X[:90]
trainY = y[:90]
testX = X[90:]

reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
reg.fit(trainX, trainY)
pred = reg.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/gradient_boosting_regressor.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': pred.tolist()
    }, f)
