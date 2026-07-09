from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
import json, os

X, y = make_classification(
    n_samples=180,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=3,
    random_state=0,
)
trainX = X[:130]
trainY = y[:130]
testX = X[130:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
clf.fit(trainX, trainY)
pred = clf.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/gradient_boosting_classifier_multi.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': pred.tolist()
    }, f)
