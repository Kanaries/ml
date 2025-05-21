import numpy as np
from sklearn.linear_model import LinearRegression
import json, os

np.random.seed(0)
X = np.random.rand(30, 2)
coef = np.array([2.0, -3.0])
y = X.dot(coef) + 5.0
model = LinearRegression().fit(X, y)
X_test = np.random.rand(10, 2)
pred = model.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/linear_regression.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
