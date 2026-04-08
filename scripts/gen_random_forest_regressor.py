from sklearn.ensemble import RandomForestRegressor
import numpy as np
import json, os

rng = np.random.RandomState(0)
X = rng.uniform(-3, 3, size=(120, 1))
y = 4 * X[:, 0] ** 2 - 2 * X[:, 0] + 1
trainX = X[:90]
trainY = y[:90]
testX = X[90:]

reg = RandomForestRegressor(n_estimators=25, random_state=0, max_features=1.0)
reg.fit(trainX, trainY)
pred = reg.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/random_forest_regressor.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': pred.tolist()
    }, f)
