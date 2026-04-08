from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import json, os

X, y = make_classification(
    n_samples=120,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=0,
)
trainX = X[:90]
trainY = y[:90]
testX = X[90:]

clf = RandomForestClassifier(n_estimators=25, random_state=0, max_features='sqrt')
clf.fit(trainX, trainY)
pred = clf.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/random_forest_classifier.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': pred.tolist()
    }, f)
