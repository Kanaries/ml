from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
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

# SAMME with depth-1 trees is the (only) algorithm in current sklearn
clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)
clf.fit(trainX, trainY)
pred = clf.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/adaboost_classifier_multi.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': pred.tolist()
    }, f)
