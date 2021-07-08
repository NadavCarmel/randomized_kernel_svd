import pickle
import numpy as np

X = np.random.randn(100, 3)

with open('../data/randn_100_3.pkl', 'wb') as f:
    pickle.dump(X, f)

