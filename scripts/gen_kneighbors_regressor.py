from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import json, os

rng = np.random.RandomState(0)
X = rng.uniform(-2, 2, size=(40, 2))
y = 3 * X[:, 0] - 2 * X[:, 1] + 1
X_test = rng.uniform(-1.5, 1.5, size=(10, 2))
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X, y)
pred = reg.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/kneighbors_regressor.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
