from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
import json, os

X, y = make_classification(
    n_samples=150,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    random_state=0,
)
trainX = X[:110]
trainY = y[:110]
testX = X[110:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
clf.fit(trainX, trainY)
pred = clf.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/gradient_boosting_classifier.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': pred.tolist()
    }, f)
