from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import json, os

X, y = make_regression(n_samples=60, n_features=4, n_informative=2, random_state=0, noise=0.1)
regr = AdaBoostRegressor(random_state=0, n_estimators=50, estimator=DecisionTreeRegressor(max_depth=3))
regr.fit(X, y)
X_test, _ = make_regression(n_samples=10, n_features=4, n_informative=2, random_state=1, noise=0.1)
pred = regr.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/ada_boost_regressor.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
