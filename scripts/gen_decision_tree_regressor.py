from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import numpy as np, json, os

X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=0)
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X, y)
X_test, _ = make_regression(n_samples=10, n_features=3, noise=0.1, random_state=1)
pred = reg.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/decision_tree_regressor.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
