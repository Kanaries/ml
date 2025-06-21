import numpy as np
from sklearn.neural_network import BernoulliRBM
import json, os

rng = np.random.RandomState(0)
X = rng.randint(2, size=(50, 6))
rbm = BernoulliRBM(n_components=2, learning_rate=0.1, batch_size=10, n_iter=20, random_state=0)
rbm.fit(X)
X_test = rng.randint(2, size=(10, 6))
trans = rbm.transform(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/bernoulli_rbm.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'X_test': X_test.tolist(),
        'expected': trans.tolist()
    }, f)
