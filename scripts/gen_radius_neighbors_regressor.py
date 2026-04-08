from sklearn.neighbors import RadiusNeighborsRegressor
import numpy as np
import json, os

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [5.0, 5.0],
    [5.0, 6.0],
    [6.0, 5.0],
], dtype=float)
y = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
X_test = np.array([
    [0.2, 0.3],
    [5.2, 5.1],
    [0.7, 0.1],
    [5.4, 5.8],
], dtype=float)
reg = RadiusNeighborsRegressor(radius=1.6)
reg.fit(X, y)
pred = reg.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/radius_neighbors_regressor.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
