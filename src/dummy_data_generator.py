import pickle
import numpy as np

X = np.random.randn(10000, 3)

with open('../data/randn_10000_3.pkl', 'wb') as f:
    pickle.dump(X, f)

